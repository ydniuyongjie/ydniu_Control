import argparse
import logging
import math
import numpy as np
import os
import cv2
import os.path as osp
import torch
from tqdm import tqdm
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. SSIM and PSNR metrics will be disabled.")
    SSIM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: LPIPS not available. Perceptual loss metric will be disabled.")
    LPIPS_AVAILABLE = False
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.data.dataset_depth import DepthDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config as load_sd_model_from_config
# 添加wandb导入
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
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
    os.makedirs(osp.join(path, 'result_ckpt'), exist_ok=True)
    os.makedirs(osp.join(path, 'models'), exist_ok=True)
    os.makedirs(osp.join(path, 'training_states'), exist_ok=True)
    os.makedirs(osp.join(path, 'visualization'), exist_ok=True)

def load_resume_state(opt):
    resume_state_path = None
    resume_ckpt_path = None
    best_ckpt_path = None

    if opt.auto_resume:
        state_path = osp.join('experiments', opt.instance_name, 'training_states')
        ckpt_path = osp.join('experiments', opt.instance_name, 'models')
        result_ckpt_path = osp.join('experiments', opt.instance_name, 'result_ckpt')

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

        # 检查是否有最佳模型文件
        if osp.isdir(result_ckpt_path):
            best_ckpt_files = list(scandir(result_ckpt_path, suffix='pth', recursive=False, full_path=False))
            if 'model_ad_best.pth' in best_ckpt_files:
                best_ckpt_path = osp.join(result_ckpt_path, 'model_ad_best.pth')

    if resume_state_path is None:
        resume_state = None
    if resume_ckpt_path is None:
        resume_ckpt = None
    if best_ckpt_path is None:
        best_ckpt = None
    else:
        device_id = torch.cuda.current_device()
        if resume_state_path is not None:
            resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id), weights_only=False)
        else:
            resume_state = None

        if resume_ckpt_path is not None:
            resume_ckpt = torch.load(resume_ckpt_path, map_location=lambda storage, loc: storage.cuda(device_id), weights_only=False)
        else:
            resume_ckpt = None

        if best_ckpt_path is not None:
            best_ckpt = torch.load(best_ckpt_path, map_location=lambda storage, loc: storage.cuda(device_id), weights_only=False)
        else:
            best_ckpt = None
        # check_resume(opt, resume_state['iter'])
    return resume_state, resume_ckpt, best_ckpt

def calculate_image_quality_metrics(generated_img, target_img):
    """
    计算生成图像与目标图像的质量指标

    图像质量指标说明：
    - SSIM (结构相似性指数): 范围[-1,1]，越接近1表示结构相似性越好
    - PSNR (峰值信噪比): 单位dB，越高表示图像质量越好，通常30dB以上为较好
    - LPIPS (感知相似性): 范围[0,1]，越接近0表示感知上越相似
    - MSE (均方误差): 越低越好，表示像素级别的差异
    - MAE (平均绝对误差): 越低越好，表示像素级别的平均差异

    Args:
        generated_img: 生成的图像 (numpy array, HWC, 0-255)
        target_img: 目标图像 (numpy array, HWC, 0-255)

    Returns:
        metrics: 包含各种质量指标的字典
    """
    metrics = {}

    try:
        # 确保图像尺寸一致
        if generated_img.shape != target_img.shape:
            target_img = cv2.resize(target_img, (generated_img.shape[1], generated_img.shape[0]))

        # 计算SSIM (结构相似性指数)
        # 范围[-1,1]，越接近1表示生成图像与目标图像在结构上越相似
        # 0.1405是一个较低的值，说明结构相似性较差
        if SSIM_AVAILABLE:
            ssim_score = ssim(generated_img, target_img, multichannel=True, channel_axis=2, data_range=255)
            metrics['ssim'] = float(ssim_score)

        # 计算PSNR (峰值信噪比)
        # 单位dB，值越高表示图像质量越好
        # 通常30dB以上为较好质量，9.0480是相当低的值
        if SSIM_AVAILABLE:
            psnr_score = psnr(generated_img, target_img, data_range=255)
            metrics['psnr'] = float(psnr_score)

        # 计算LPIPS (学习感知图像块相似性)
        # 基于深度学习的感知相似性，范围[0,1]
        # 值越接近0表示图像在感知上越相似
        # 0.7342表示感知相似性较差
        if LPIPS_AVAILABLE:
            try:
                # 初始化LPIPS模型（如果还没有）
                if not hasattr(calculate_image_quality_metrics, 'lpips_model'):
                    calculate_image_quality_metrics.lpips_model = lpips.LPIPS(net='alex').cuda()

                # 转换图像格式为LPIPS所需的格式
                gen_tensor = torch.from_numpy(generated_img).float().cuda() / 255.0
                target_tensor = torch.from_numpy(target_img).float().cuda() / 255.0

                # 从HWC转为NCHW，并归一化到[-1,1]
                gen_tensor = gen_tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1

                with torch.no_grad():
                    lpips_score = calculate_image_quality_metrics.lpips_model(gen_tensor, target_tensor)
                metrics['lpips'] = float(lpips_score.item())
            except Exception as e:
                print(f"Warning: LPIPS calculation failed: {e}")
                metrics['lpips'] = None

        # 计算MSE (均方误差)
        # 衡量预测值与真实值之间差异的平均值
        # 值越小越好，8096.1872是一个相当大的误差值
        mse = np.mean((generated_img.astype(float) - target_img.astype(float)) ** 2)
        metrics['mse'] = float(mse)

        # 计算MAE (平均绝对误差)
        # 预测值与真实值之间绝对差异的平均值
        # 值越小越好，79.6689也是较大的误差值
        mae = np.mean(np.abs(generated_img.astype(float) - target_img.astype(float)))
        metrics['mae'] = float(mae)

    except Exception as e:
        print(f"Error calculating image quality metrics: {e}")
        # 返回默认值
        metrics = {'ssim': None, 'psnr': None, 'lpips': None, 'mse': None, 'mae': None}

    return metrics

def calculate_composite_score(metrics):
    """
    计算综合评分，用于早停判断

    综合评分说明：
    - 综合评分是多个指标的加权组合，用于模型选择和保存
    - 值越高表示综合表现越好
    - 例如：0.1360 表示当前模型的综合表现

    Args:
        metrics: 包含各种质量指标的字典

    Returns:
        composite_score: 综合评分（越高越好）
    """
    score = 0.0
    count = 0

    # 检查可用的指标
    has_ssim = metrics.get('ssim') is not None
    has_psnr = metrics.get('psnr') is not None
    has_lpips = metrics.get('lpips') is not None

    # 根据可用指标动态分配权重
    if has_ssim and has_psnr and has_lpips:
        # 全部指标可用 - 原始权重分配
        ssim_weight = 0.4
        psnr_weight = 0.3
        lpips_weight = 0.3
    elif has_ssim and has_psnr:
        # 缺少LPIPS - 重新分配权重
        ssim_weight = 0.6
        psnr_weight = 0.4
        lpips_weight = 0.0
    elif has_ssim and has_lpips:
        # 缺少PSNR - 重新分配权重
        ssim_weight = 0.7
        psnr_weight = 0.0
        lpips_weight = 0.3
    elif has_psnr and has_lpips:
        # 缺少SSIM - 重新分配权重
        ssim_weight = 0.0
        psnr_weight = 0.6
        lpips_weight = 0.4
    else:
        # 只有一个或没有高级指标
        ssim_weight = 0.0
        psnr_weight = 0.0
        lpips_weight = 0.0

    # SSIM权重最高（0-1，越高越好）
    if has_ssim:
        score += metrics['ssim'] * ssim_weight
        count += ssim_weight

    # PSNR权重（通常20-40，越高越好，需要归一化）
    if has_psnr:
        normalized_psnr = min(max(metrics['psnr'] - 20, 0) / 20, 1)  # 归一化到[0,1]
        score += normalized_psnr * psnr_weight
        count += psnr_weight

    # LPIPS权重（0-1，越低越好）
    if has_lpips:
        score += (1 - metrics['lpips']) * lpips_weight  # 反转，越高越好
        count += lpips_weight

    # 如果没有高级指标，使用MSE的倒数作为简单评分
    if count == 0 and metrics.get('mse') is not None:
        score = 1 / (1 + metrics['mse'])  # MSE越低，分数越高
        count = 1

    return score if count > 0 else 0.0



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bsize",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--val_iter",
        type=int,
        default=2000,
        help="validation frequency"
        )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="Frequency of training information output",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
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
        action='store_true',
        help="resume training if last checkpoint is available",
        )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
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
        default="configs/stable-diffusion/sd-v1-train.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_depth",
        help="experiment name",
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
        "--sample_steps",
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
        default='ydyongjieniu',
        help='wandb entity name'
)
    parser.add_argument(
        "--instance_name",
        type=str,
        default="default",
        help="Name of the instance in which the program runs, 'e.g.' prior_lora_cat_100_1(prior_adapter_obj_classnum_batch)",
    )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
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
    train_dataset = DepthDataset('depth/depth.txt')
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle=True,#(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True
        )#sampler=train_sampler

    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)
    experiments_root = osp.join('experiments', opt.instance_name)
    mkdir_and_rename(experiments_root)
    #验证数据
    val_data = train_dataset[14]
    with torch.no_grad():
        # 将张量转换为numpy数组
        im_np = val_data['im'].cpu().detach().numpy()
        # 从[0,1]范围转换到[0,255]范围
        im_np = (im_np * 255).astype(np.uint8)
        # im_np = im_np.squeeze(0)
        # 从[C,H,W]转换为[H,W,C]
        im_np = im_np.transpose(1, 2, 0)
        # 从RGB转换回BGR（因为OpenCV默认使用BGR）
        im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        # 保存图像
        cv2.imwrite(os.path.join(experiments_root, 'visualization', 'target.jpg'), im_np)
        # 计算验证损失
        # edge = net_G(data['im'].cuda(non_blocking=True))[-1]
        # edge = edge>0.5
        # edge = edge.float()#1,1,512,512
        # im_edge = tensor2img(edge)
        # cv2.imwrite(os.path.join(experiments_root, 'visualization', 'edge_1.png'), im_edge)
        depth_data = val_data['depth'].cuda(non_blocking=True)
        im_depth = tensor2img(depth_data)
        cv2.imwrite(os.path.join(experiments_root, 'visualization', 'depth.png'), im_depth)
        c = model.get_learned_conditioning(val_data['sentence'])
        # z = model.encode_first_stage((val_data['im']*2-1.).cuda(non_blocking=True))
        # z = model.get_first_stage_encoding(z)
        features_adapter = None
        control_injectors = None
        # features_adapter = [f*0.0 if isinstance(f, torch.Tensor) else f for f in features_adapter] # features_adapter置为0.0      
        if opt.dpm_solver:
            sampler = DPMSolverSampler(model)
        elif opt.plms:
            sampler = PLMSSampler(model)
        else:
            # 使用新的采样器
            sampler = DDIMSampler(model)
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
                                            features_adapter=features_adapter,
                                            control_injectors=control_injectors)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        for id_sample, x_sample in enumerate(x_samples_ddim):
            x_sample = 255.*x_sample
            img = x_sample.astype(np.uint8)
            img = cv2.putText(img.copy(), val_data['sentence'], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imwrite(os.path.join(experiments_root, 'visualization', 'origin.png'), img[:,:,::-1])
            
            # 如果启用了wandb，则将生成的图像记录到wandb
            if opt.use_wandb and wandb_available:
                # 将生成的图像转换为wandb.Image格式
                wandb_image = wandb.Image(
                    img, 
                    caption=f"origin image from model")
                wandb.log({
                    f"val/origin image": wandb_image
                }, step=0)

    # depth encoder
    model_ad = Adapter(cin=3 * 64, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)

    # optimizer
    params = list(model_ad.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])
    
    # 添加CosineAnnealingLR学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=opt.epochs, 
        eta_min=1e-7  # 最小学习率
    )

    # 重新启用早停机制，基于图像质量评估
    best_val_score = -float('inf')  # 越大越好（SSIM、PSNR）
    worst_lpips = float('inf')      # 越小越好（LPIPS）
    patience = 5  # 早停耐心值
    patience_counter = 0  # 早停计数器
    best_model_state = None  # 最佳模型状态

    # 早停计算说明:
    # - 数据集: 12000张图片, 批量大小: 6, 每epoch: 2000次迭代
    # - 验证频率: 每2000次迭代 (每epoch验证一次)
    # - 早停条件: 连续5次验证无改善 = 连续5个epoch无改善



    # resume state
    resume_state, resume_ckpt, best_ckpt = load_resume_state(opt)
    if resume_state is None or resume_ckpt is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.instance_name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        # logger.info(dict2str(config))
    if resume_state is not None and resume_ckpt is not None :
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.instance_name}_{get_time_str()}.log")
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

        # 注释：暂时禁用早停相关参数的恢复
        # if best_ckpt is not None:
        #     # 如果有最佳模型，加载为最佳模型状态
        #     best_model_state = best_ckpt
        #     logger.info("Loaded best model state for early stopping")
        #     # 设置一个合理的初始值，实际中应该保存和加载这个值
        #     best_val_loss = 0.1
        #     logger.info(f"Set initial best_val_loss for resumed training: {best_val_loss}")
        # else:
        #     logger.info("No best model found, starting with fresh early stopping parameters")

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)
    
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

    # 训练循环，支持早停
    early_stop = False
    for epoch in range(start_epoch, opt.epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        # train
        gen_image_count=0
        for batch_idx, data in enumerate(train_dataloader):
            if epoch == start_epoch and batch_idx < start_iter:
                continue
                        
            with torch.no_grad():
                depth_data = data['depth'].cuda(non_blocking=True)
                c = model.get_learned_conditioning(data['sentence'])
                z = model.encode_first_stage((data['im']*2-1.).cuda(non_blocking=True))                
                z = model.get_first_stage_encoding(z)

            optimizer.zero_grad()
            model.zero_grad()            
            features_adapter = model_ad(depth_data)
            l_pixel, loss_dict = model(z, c=c, features_adapter=features_adapter)
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
            if ((current_iter + 1) % config['training']['save_freq'] == 0):
                save_filename = f'model_ad_{current_iter + 1}.pth'
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
            if (current_iter+1)% opt.val_iter == 0:  # Always run validation for single GPU training
                gen_image_count+=1

                val_data = train_dataset[14]

                with torch.no_grad():
                    depth_data = val_data['depth'].cuda(non_blocking=True)
                    c = model.get_learned_conditioning(val_data['sentence'])
                    # 处理深度数据（类似sketch中的edge处理）
                    depth_data=depth_data.unsqueeze(0)
                    features_adapter = model_ad(depth_data)

                    if opt.dpm_solver:
                        sampler = DPMSolverSampler(model)
                    elif opt.plms:
                        sampler = PLMSSampler(model)
                    else:
                        sampler = DDIMSampler(model)
                    logger.info(f"Validation image shape: {val_data['im'].shape}")

                    # 保存深度图像用于可视化
                    depth_data_vis = val_data['depth'].cuda(non_blocking=True)
                    im_depth_vis = tensor2img(depth_data_vis)
                    cv2.imwrite(os.path.join(experiments_root, 'visualization', 'depth_%04d.png'%epoch), im_depth_vis)

                    # 如果启用了wandb，则将深度图像记录到wandb
                    if opt.use_wandb and wandb_available:
                        # 将深度图像转换为wandb.Image格式
                        wandb_depth_image = wandb.Image(
                            im_depth_vis,
                            caption=f"Depth image at epoch {epoch}"
                        )
                        wandb.log({
                            f"val/depth_image_e{epoch:04d}": wandb_depth_image
                        }, step=current_iter)
                    # 从随机噪声生成图像（与sketch验证逻辑一致）
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=model.get_learned_conditioning(opt.n_samples * [""]),
                                                        eta=opt.ddim_eta,
                                                        x_T=None,  # 从随机噪声开始
                                                        features_adapter=features_adapter)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    # 计算生成图像与目标图像的质量指标
                    target_img = tensor2img(val_data['im'])  # 获取目标图像

                    for id_sample, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255.*x_sample
                        generated_img = x_sample.astype(np.uint8)

                        # 计算图像质量指标
                        metrics = calculate_image_quality_metrics(generated_img, target_img)

                        # 记录指标到日志
                        metrics_str = ", ".join([f"{k}: {v:.4f}" if v is not None else f"{k}: N/A"
                                                for k, v in metrics.items()])
                        logger.info(f"Validation metrics - {metrics_str}")
                        logger.info("指标解读: SSIM>0.5较好, PSNR>30dB较好, LPIPS<0.2较好, MSE/MAE越低越好")

                        # 记录到wandb
                        if opt.use_wandb and wandb_available:
                            wandb_metrics = {}
                            for k, v in metrics.items():
                                if v is not None:
                                    wandb_metrics[f"val/{k}"] = v
                            wandb.log(wandb_metrics, step=current_iter)

                        # 计算综合评分
                        composite_score = calculate_composite_score(metrics)

                        # 早停判断和最佳模型保存
                        # 早停机制：每次验证时判断，如果连续5次验证没有改善则停止训练
                        if composite_score > best_val_score:
                            best_val_score = composite_score
                            patience_counter = 0
                            best_model_state = model_ad.state_dict().copy()

                            # 保存最佳模型
                            save_filename = 'model_ad_best.pth'
                            save_path = os.path.join(experiments_root, 'result_ckpt', save_filename)
                            save_dict = {}
                            for key, param in best_model_state.items():
                                save_dict[key] = param.cpu()
                            torch.save(save_dict, save_path)

                            logger.info(f"New best model saved with composite score: {composite_score:.4f}")
                            logger.info(f"当前最佳综合评分: {composite_score:.4f} - 用于判断模型性能的综合指标")

                            # 记录到wandb
                            if opt.use_wandb and wandb_available:
                                wandb.log({"val/composite_score": composite_score}, step=current_iter)
                        else:
                            # 性能没有改善，增加早停计数器
                            patience_counter += 1
                            logger.info(f"Current composite score: {composite_score:.4f}, Best: {best_val_score:.4f}")
                            logger.info(f"Validation score did not improve. Patience counter: {patience_counter}/{patience}")
                            logger.info(f"早停说明: 连续{patience}次验证无改善将自动停止训练，当前第{patience_counter}次")

                        # 检查是否需要早停
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered after {patience} validations without improvement")
                            logger.info(f"Best composite score: {best_val_score:.4f}")
                            logger.info(f"训练已早停: 在连续{patience}次验证（约{patience * opt.val_iter}次迭代）中性能未提升")
                            early_stop = True
                            break

                        # 只对第一个生成的样本进行后续处理（保存图像等）
                        if id_sample == 0:
                            img = cv2.putText(generated_img.copy(), val_data['sentence'], (10,30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            cv2.imwrite(os.path.join(experiments_root, 'visualization', 'sample_e%04d_s%04d.png'%(epoch, gen_image_count)), img[:,:,::-1])

                            # 如果启用了wandb，则将生成的图像记录到wandb
                            if opt.use_wandb and wandb_available:
                                wandb_image = wandb.Image(
                                    img,
                                    caption=f"Epoch {epoch} Sample {gen_image_count}: {val_data['sentence']}"
                                )
                                wandb.log({
                                    f"val/generated_image_e{epoch:04d}_s{gen_image_count:04d}": wandb_image
                                }, step=current_iter)

                    logger.info(f"Validation completed at epoch {epoch}, iter {current_iter+1}")  
    
            
            # 正常训练流程
            current_iter += 1
        # 检查是否早停，如果早停则跳出epoch循环
        if early_stop:
            break

        # 每个epoch结束时更新学习率调度器并记录当前学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch} finished. Current learning rate: {current_lr}')
    
    # 关闭进度条
    progress_bar.close()

    # 检查是否早停
    if patience_counter >= patience:
        logger.info("Training stopped early due to patience limit reached")
        # 使用最佳模型状态恢复model_ad
        if best_model_state is not None:
            model_ad.load_state_dict(best_model_state)
            logger.info("Restored best model state for final saving")

    # 保存最终模型
    if best_model_state is not None:
        # 如果有最佳模型，保存最佳模型作为最终模型
        save_filename = 'model_ad_final_best.pth'
        save_path = os.path.join(experiments_root, 'result_ckpt', save_filename)
        save_dict = {}
        for key, param in best_model_state.items():
            save_dict[key] = param.cpu()
        torch.save(save_dict, save_path)
        logger.info(f"Saved best model as final model with composite score: {best_val_score:.4f}")
    else:
        # 如果没有最佳模型（没有验证过），保存当前模型
        save_filename = 'model_ad_final.pth'
        save_path = os.path.join(experiments_root, 'result_ckpt', save_filename)
        state_dict = model_ad.state_dict()
        save_dict = {}
        for key, param in state_dict.items():
            save_dict[key] = param.cpu()
        torch.save(save_dict, save_path)
        logger.info(f"Saved final model at the end of training")
    # 结束wandb会话
    if opt.use_wandb and wandb_available:
        wandb.finish()


if __name__ == '__main__':
    main()
