import argparse
import logging
import math
import os
import os.path as osp
import time
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
from PIL import Image

from ldm.data.dataset_coco import dataset_coco_mask_color
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.model_edge import pidinet

# 添加wandb导入
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu",weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    # if osp.exists(path):
    #     new_name = path + '_archived_' + get_time_str()
    #     print(f'Path already exists. Rename it to {new_name}', flush=True)
    #     os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'), exist_ok=True)
    os.makedirs(osp.join(path, 'training_states'), exist_ok=True)
    os.makedirs(osp.join(path, 'visualization'), exist_ok=True)

def load_resume_state(opt):
    resume_state_path = None
    resume_ckpt_path =None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        ckpt_path = osp.join('experiments', opt.name, 'models')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
        if osp.isdir(ckpt_path):
            ckpts = list(scandir(ckpt_path, suffix='pth', recursive=False, full_path=False))
            if len(ckpts) != 0:
                ckpts = [float(v.split('.pth')[0].rsplit('_',1)[-1]) for v in ckpts]
                resume_ckpt_path = osp.join(ckpt_path, f'model_ad_{max(ckpts):.0f}.pth')
    if resume_state_path is None:
        resume_state = None
    if resume_ckpt_path is None:
        resume_ckpt = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        resume_ckpt= torch.load(resume_ckpt_path, map_location=lambda storage, loc: storage.cuda(device_id))
        # check_resume(opt, resume_state['iter'])
    return resume_state,resume_ckpt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=2,
    help="Batch size during training"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=6,
    help="Epochs during training"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="the prompt to render"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="the prompt to render"
)
parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling,DPM (Diffusion Probabilistic Models) Solver 是一种用于扩散模型采样的快速算法",
)
parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--auto_resume",
        default=True,
        help="resume training if last checkpoint is available",
)
parser.add_argument(
        "--ckpt",
        type=str,
        default="models/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
)
parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/train_sketch.yaml",
        help="path to config which constructs model",
)
parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="Frequency of training information output",
)
parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
)
parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
        "--gpus",
        default=[0],
        help="gpu idx",
)
parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='enable wandb logging'
)
parser.add_argument(
        '--wandb_project',
        type=str,
        default='t2i-adapter',
        help='wandb project name'
)
parser.add_argument(
        '--wandb_entity',
        type=str,
        default='ydniuyongjie',
        help='wandb entity name'
)
parser.add_argument(
        "--instance_name",
        type=str,
        default="default",
        help="Name of the instance in which the program runs, 'e.g.' prior_lora_cat_100_1(prior_adapter_obj_classnum_batch)",
    )
# parser.add_argument(
#         '--wandb_run_id',
#         type=str,
#         default='None',
#         help='wandb run id (if not provided, wandb will generate a random one)'
# )
opt = parser.parse_args()

if __name__ == '__main__':
    # torch.manual_seed(42)
    # random.seed(42)
    # np.random.seed(42)
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']

    # single GPU setting
    torch.backends.cudnn.benchmark = True
    device='cuda'

    # 初始化wandb
    if opt.use_wandb and wandb_available:
        # 如果指定了wandb_run_id，则使用固定的ID，否则使用默认的随机ID
        if hasattr(opt, 'wandb_run_id') and opt.wandb_run_id:
            wandb.init(
                project=opt.wandb_project,
                entity=opt.wandb_entity,
                name=opt.instance_name,
                id=opt.wandb_run_id,
                config={
                    "batch_size": opt.bsize,
                    "epochs": opt.epochs,
                    "learning_rate": config['training']['lr'],                   
                }
            )
        else:
            wandb.init(
                project=opt.wandb_project,
                entity=opt.wandb_entity,
                name=opt.instance_name,
                config={
                    "batch_size": opt.bsize,
                    "epochs": opt.epochs,
                    "learning_rate": config['training']['lr'],                    
                }
            )

    # dataset
    path_json_train = 'coco_stuff/mask/annotations/captions_train2017.json'
    path_json_val = 'coco_stuff/mask/annotations/captions_val2017.json'
    train_dataset = dataset_coco_mask_color(path_json_train,
    root_path_im='coco/train2017',
    # root_path_mask='coco_stuff/mask/train2017_color',
    image_size=512
    )
    val_dataset = dataset_coco_mask_color(path_json_val,
    root_path_im='coco/val2017',
    # root_path_mask='coco_stuff/mask/val2017_color',
    image_size=512
    )
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.bsize,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)

    # edge_generator
    net_G = pidinet()
    ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
    net_G.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
    net_G.cuda()

    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)

    # sketch encoder
    model_ad = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)

    # optimizer
    params = list(model_ad.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state,resume_ckpt = load_resume_state(opt)
    if resume_state is None or resume_ckpt is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        # logger.info(dict2str(config))
    if resume_state is not None and resume_ckpt is not None :
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        # logger.info(dict2str(config))
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")   
        
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter'] # 实际迭代次数从恢复点开始计数        
        # 加载优化器状态
        optimizer.load_state_dict(resume_state['optimizers'])
        logger.info("Training has resumed.So loaded optimizer state")
        model_ad.load_state_dict(resume_ckpt)
        logger.info("Training has resumed.So Loaded model_ad state")

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    # 计算总批次数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)) #math.ceil(500)

    
    # 显示训练信息
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.epochs}")
    logger.info(f"  Instantaneous batch size per device = {opt.bsize}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {opt.bsize}")
    logger.info(f"  Total optimization steps = {num_update_steps_per_epoch * opt.epochs}")

    
    # 初始化进度条
    total_steps = num_update_steps_per_epoch * opt.epochs
    progress_bar = tqdm(
        range(0, total_steps),
        desc="Steps",
    )
    
    # 如果是恢复训练，更新进度条的初始位置
    if current_iter > 0:
        progress_bar.update(current_iter)
        logger.info(f"Resuming training from iteration {current_iter}. Total iterations: {total_steps}. Remaining iterations: {total_steps - current_iter}")
    else:
        logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}. Total iterations: {total_steps}')

    
    # training    
    start_iter= current_iter - start_epoch * num_update_steps_per_epoch
    for epoch in range(start_epoch, opt.epochs):
        # train
        # from itertools import islice
        
        for batch_idx, data in enumerate(train_dataloader):#enumerate(islice(train_dataloader, 500)):
            # Skip batches if resuming from a specific iteration
            # 跳过已完成的迭代（关键！恢复时避免重复训练）
            if epoch == start_epoch and batch_idx < start_iter:
                continue
                
            # 正常训练流程
            current_iter += 1
            with torch.no_grad():
                edge = net_G(data['im'].cuda(non_blocking=True))[-1]
                edge = edge>0.5
                edge = edge.float()
                c = model.get_learned_conditioning(data['sentence'])
                z = model.encode_first_stage((data['im']*2-1.).cuda(non_blocking=True))
                z = model.get_first_stage_encoding(z)

            optimizer.zero_grad()
            model.zero_grad()
            features_adapter = model_ad(edge)
            l_pixel, loss_dict = model(z, c=c, features_adapter = features_adapter)
            l_pixel.backward()
            optimizer.step()

            # 更新进度条
            progress_bar.update(1)
            
            if (current_iter+1)%opt.print_fq == 0:
                # 记录简洁的训练损失到日志文件
                clean_loss_dict = {}
                for key, value in loss_dict.items():
                    if hasattr(value, 'item'):
                        clean_loss_dict[key] = round(value.item(), 6)
                    else:
                        clean_loss_dict[key] = round(value, 6)
                logger.info(clean_loss_dict)
                
                # 记录到wandb
                if opt.use_wandb and wandb_available:
                    # 将损失字典中的键名转换为wandb友好的格式
                    wandb_loss_dict = {}
                    for key, value in loss_dict.items():
                        wandb_loss_dict[f"train/{key}"] = value.item() if hasattr(value, 'item') else value
                    wandb.log(wandb_loss_dict, step=current_iter)
                # 更新进度条显示
                progress_bar.set_postfix(**clean_loss_dict)

            # save checkpoint
            if (current_iter+1)%config['training']['save_freq'] == 0:
                save_filename = f'model_ad_{current_iter+1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                state_dict = model_ad.state_dict()
                save_dict = {}
                for key, param in state_dict.items():
                    save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)
            # save state
                state = {'epoch': epoch, 
                         'iter': current_iter+1, 
                         'optimizers': optimizer.state_dict()
                         }
                save_filename = f'{current_iter+1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)
                
                # 记录检查点保存信息到日志
                logger.info(f"Saved checkpoint at epoch {epoch}, iter {current_iter+1}")                
               
        # val
        if True:  # Always run validation for single GPU training
            # 初始化验证损失累积变量
            val_loss_simple = 0.0
            val_loss_vlb = 0.0
            val_loss_total = 0.0
                        
            for data in val_dataloader:
                with torch.no_grad():
                    # 计算验证损失
                    edge = net_G(data['im'].cuda(non_blocking=True))[-1]
                    edge = edge>0.5
                    edge = edge.float()
                    c = model.get_learned_conditioning(data['sentence'])
                    z = model.encode_first_stage((data['im']*2-1.).cuda(non_blocking=True))
                    z = model.get_first_stage_encoding(z)
                    features_adapter = model_ad(edge)
                    
                    # 计算验证损失
                    val_loss, val_loss_dict = model(z, c=c, features_adapter=features_adapter)
                    
                    # 累积验证损失
                    val_loss_simple = val_loss_dict.get('loss_simple', val_loss).item()
                    val_loss_vlb = val_loss_dict.get('loss_vlb', torch.tensor(0.0)).item()
                    val_loss_total = val_loss.item()
                    
                    if opt.dpm_solver:
                        sampler = DPMSolverSampler(model)
                    elif opt.plms:
                        sampler = PLMSSampler(model)
                    else:
                        sampler = DDIMSampler(model)
                    print(data['im'].shape)
                    c = model.get_learned_conditioning(data['sentence'])
                    edge = net_G(data['im'].cuda(non_blocking=True))[-1]
                    edge = edge>0.5
                    edge = edge.float()
                    im_edge = tensor2img(edge)
                    cv2.imwrite(os.path.join(experiments_root, 'visualization', 'edge_%04d.png'%epoch), im_edge)
                    
                    # 如果启用了wandb，则将边缘图像记录到wandb
                    if opt.use_wandb and wandb_available:
                        # 将边缘图像转换为wandb.Image格式
                        wandb_edge_image = wandb.Image(
                            im_edge, 
                            caption=f"Edge image at epoch {epoch}"
                        )
                        wandb.log({
                            f"val/edge_image_e{epoch:04d}": wandb_edge_image
                        }, step=current_iter)
                    features_adapter = model_ad(edge)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=model.get_learned_conditioning(opt.n_samples * [""]),
                                                        eta=opt.ddim_eta,
                                                        x_T=None,
                                                        features_adapter=features_adapter)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    for id_sample, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255.*x_sample
                        img = x_sample.astype(np.uint8)
                        img = cv2.putText(img.copy(), data['sentence'][0], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.imwrite(os.path.join(experiments_root, 'visualization', 'sample_e%04d_s%04d.png'%(epoch, id_sample)), img[:,:,::-1])
                        
                        # 如果启用了wandb，则将生成的图像记录到wandb
                        if opt.use_wandb and wandb_available:
                            # 将生成的图像转换为wandb.Image格式
                            wandb_image = wandb.Image(
                                img[:, :, ::-1], 
                                caption=f"Epoch {epoch} Sample {id_sample}: {data['sentence'][0]}"
                            )
                            wandb.log({
                                f"val/generated_image_e{epoch:04d}_s{id_sample:04d}": wandb_image
                            }, step=current_iter)
                    break
    # 关闭进度条
    progress_bar.close()
    # 保存模型
    # 添加最终模型保存代码（位置1）
    save_filename = f'model_ad_final.pth'
    save_path = os.path.join(experiments_root, 'models', save_filename)
    state_dict = model_ad.state_dict()
    save_dict = {}
    for key, param in state_dict.items():
        save_dict[key] = param.cpu()
    torch.save(save_dict, save_path)
    logger.info(f"Saved final model at the end of training")
    # 结束wandb会话
    if opt.use_wandb and wandb_available:
        wandb.finish()
            
