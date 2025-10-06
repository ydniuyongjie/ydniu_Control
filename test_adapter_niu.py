import os
import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (
    diffusion_inference_niu, get_adapters, get_base_argument_parser, get_sd_models_niu
)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (
    ExtraCondition, get_adapter_feature, get_cond_model
)
from ldm.modules.diffusionmodules.ControlInjectionBlock import ControlInjectionBlock

def get_control_injectors(opt, channels, device):
    # 加载ControlInjectionBlock权重
    control_ckpt = opt.control_ckpt
    control_injectors = torch.nn.ModuleList(
        ControlInjectionBlock(channels=ch, time_emb_dim=1280) for ch in channels
    ).to(device)
    if os.path.exists(control_ckpt):
        control_injectors.load_state_dict(torch.load(control_ckpt, map_location=device))
    control_injectors.eval()
    return control_injectors

def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument('--which_cond', type=str, required=True, choices=supported_cond)
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--control_ckpt', type=str, required=True, help='ControlInjectionBlock权重路径')
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{opt.condition}'
    os.makedirs(opt.outdir, exist_ok=True)
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 支持单图和批量
    if opt.prompt.endswith('.txt'):
        image_paths, prompts = [], []
        with open(opt.prompt, 'r') as f:
            for line in f:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]

    # 模型准备
    sd_model, sampler = get_sd_models_niu(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = None
    if opt.cond_inp_type == 'image':
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))
    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    # ControlInjectionBlock准备
    adapter_channels = [320, 640, 1280, 1280][:4]
    control_injectors = get_control_injectors(opt, adapter_channels, opt.device)

    # 推理
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)
                cv2.imwrite(os.path.join(opt.outdir, f'{v_idx:05}_{which_cond}.png'), tensor2img(cond))

                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                
                opt.prompt = prompt
                result = diffusion_inference_niu(
                    opt, sd_model, sampler, adapter_features,control_injectors, append_to_context
                    
                )
                cv2.imwrite(os.path.join(opt.outdir, f'{v_idx:05}_result.png'), tensor2img(result))
                # 不用adapter/control的原生结果
                result = diffusion_inference_niu(
                    opt, sd_model, sampler, None, None,append_to_context                    
                )
                cv2.imwrite(os.path.join(opt.outdir, f'{v_idx:05}_origin.png'), tensor2img(result))

if __name__ == '__main__':
    main()