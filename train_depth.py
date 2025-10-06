import argparse
import logging
import math
import numpy as np
import os
import cv2
import os.path as osp
import torch
from tqdm import tqdm
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.data.dataset_depth import DepthDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config
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
    os.makedirs(osp.join(path, 'result_ckpt'), exist_ok=True)
    os.makedirs(osp.join(path, 'models'), exist_ok=True)
    os.makedirs(osp.join(path, 'training_states'), exist_ok=True)
    os.makedirs(osp.join(path, 'visualization'), exist_ok=True)

def load_resume_state(opt):
    resume_state_path = None
    resume_ckpt_path =None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.instance_name, 'training_states')
        ckpt_path = osp.join('experiments', opt.instance_name, 'models')
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



def parsr_args():
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
        default=True,
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
        default='ydniuyongjie',
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
    opt = parsr_args()
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
        cv2.imwrite(os.path.join(experiments_root, 'visualization', 'traget.jpg'), im_np)
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
            img = cv2.putText(img.copy(), val_data['sentence'][0], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
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



    # resume state
    resume_state,resume_ckpt = load_resume_state(opt)
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
                # 初始化验证损失累积变量
                val_loss_simple = 0.0
                val_loss_vlb = 0.0
                val_loss_total = 0.0
                gen_image_count+=1
                
                val_data = train_dataset[14]                            

                with torch.no_grad():
                    # 计算验证损失
                    # edge = net_G(data['im'].cuda(non_blocking=True))[-1]
                    # edge = edge>0.5
                    # edge = edge.float()
                    depth_data = val_data['depth'].cuda(non_blocking=True)
                    c = model.get_learned_conditioning(val_data['sentence'])
                    #改变形状
                    val_data['im']=val_data['im'].unsqueeze(0)
                    z = model.encode_first_stage((val_data['im']*2-1.).cuda(non_blocking=True))
                    z = model.get_first_stage_encoding(z)
                    depth_data=depth_data.unsqueeze(0)
                    features_adapter = model_ad(depth_data)
                    
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
                    print(val_data['im'].shape)
                    c = model.get_learned_conditioning(val_data['sentence'])

                    depth_data = val_data['depth'].cuda(non_blocking=True)
                    im_edge = tensor2img(depth_data)
                    cv2.imwrite(os.path.join(experiments_root, 'visualization', 'depth_%04d.png'%epoch), im_edge)
                    
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
                    # features_adapter = model_ad(depth_data)
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
                        img = cv2.putText(img.copy(), val_data['sentence'][0], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.imwrite(os.path.join(experiments_root, 'visualization', 'sample_e%04d_s%04d.png'%(epoch, gen_image_count)), img[:,:,::-1])
                        
                        # 如果启用了wandb，则将生成的图像记录到wandb
                        if opt.use_wandb and wandb_available:
                            # 将生成的图像转换为wandb.Image格式
                            wandb_image = wandb.Image(
                                img, 
                                caption=f"Epoch {epoch} Sample {gen_image_count}: {val_data['sentence'][0]}"
                            )
                            wandb.log({
                                f"val/generated_image_e{epoch:04d}_s{gen_image_count:04d}": wandb_image
                            }, step=current_iter)  
            
            # 正常训练流程
            current_iter += 1            
    # 关闭进度条
    progress_bar.close()
    # 保存模型
    # 添加最终模型保存代码（位置1）
    save_filename = f'model_ad_final.pth'
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
