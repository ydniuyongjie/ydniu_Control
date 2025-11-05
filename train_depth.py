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

# 检查Depth Anything V2是否可用于深度图提取
# 重要：数据集中的深度图使用Depth Anything V2生成，验证也必须使用相同模型
try:
    import sys
    depth_anything_path = "/home/tniuyj/code/Depth-Anything-V2"
    if depth_anything_path not in sys.path:
        sys.path.insert(0, depth_anything_path)
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_V2_AVAILABLE = True
    print("✅ Depth Anything V2 可用于深度图提取验证")
    print("注意：使用与数据集生成相同的深度估计模型确保一致性")
except ImportError as e:
    print(f"❌ 无法导入Depth Anything V2用于验证: {e}")
    print("错误：无法找到与数据集匹配的深度估计模型")
    DEPTH_ANYTHING_V2_AVAILABLE = False

def initialize_depth_estimator(weight_path="depth_weight/depth_anything_v2_vitl.pth"):
    """
    初始化Depth Anything V2深度图估计器用于验证

    重要说明：
    - 数据集中的深度图使用Depth Anything V2生成
    - 验证必须使用相同的模型以确保一致性
    - 自动选择GPU/CPU模式以获得最佳性能
    """
    if not DEPTH_ANYTHING_V2_AVAILABLE:
        print("❌ Depth Anything V2不可用，无法进行深度一致性验证")
        print("   原因：数据集使用Depth Anything V2生成深度图，验证也必须使用相同模型")
        return None

    # 检查权重文件是否存在
    if not os.path.exists(weight_path):
        print(f"❌ 深度估计器权重文件不存在: {weight_path}")
        print("   请确保Depth Anything V2权重文件存在")
        return None

    try:
        print("初始化Depth Anything V2深度图估计器...")
        print(f"配置: ViT-Large encoder, 与数据集生成配置一致")
        print(f"权重: {weight_path}")

        model = DepthAnythingV2(
            encoder='vitl',  # ViT-Large，与数据集生成时使用相同的编码器
            features=256,
            out_channels=[256, 512, 1024, 1024]
        )

        # 加载权重到CPU，避免占用GPU资源
        print("加载权重文件...")
        checkpoint = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(checkpoint)

        # 使用GPU模式进行验证，提高验证效率
        if torch.cuda.is_available():
            model.cuda()
            print("   使用GPU加速深度图提取")
        else:
            model.cpu()
            print("   使用CPU进行深度图提取")

        model.eval()

        print("✅ Depth Anything V2深度估计器初始化成功")
        print("   模型将与数据集使用相同的深度估计配置，确保验证一致性")
        return model
    except Exception as e:
        print(f"❌ Depth Anything V2深度估计器初始化失败: {e}")
        print("   无法进行深度一致性验证，将回退到图像相似性验证")
        return None

def extract_depth_from_image(depth_estimator, image):
    """
    使用Depth Anything V2从生成图像中提取深度图

    重要说明：
    - 使用与数据集生成相同的Depth Anything V2模型
    - 确保验证的一致性和准确性

    Args:
        depth_estimator: Depth Anything V2深度估计模型
        image: 输入图像 (numpy array, HWC, BGR格式, 0-255)

    Returns:
        depth_map: 深度图 (numpy array, HW, 0-255)
    """
    if depth_estimator is None:
        print("⚠️ Depth Anything V2深度估计器不可用，返回空深度图")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    try:
        # 确保图像格式正确（BGR uint8格式，Depth Anything V2的输入要求）
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        # 使用Depth Anything V2生成深度图
        # 与数据集生成时使用完全相同的处理方式
        with torch.no_grad():
            depth = depth_estimator.infer_image(image)  # HxW raw depth map

        # 归一化到0-255范围，与数据集处理方式保持一致
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)

        return depth
    except Exception as e:
        print(f"❌ Depth Anything V2深度图提取失败: {e}")
        print("   将返回空深度图，验证结果可能不准确")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

def calculate_gradient(image):
    """
    计算图像的梯度

    Args:
        image: 输入图像 (numpy array, HW, 0-255)

    Returns:
        gradient: 梯度幅值 (numpy array, HW, 0-255)
    """
    try:
        # 转换为float类型并归一化到[0,1]
        img_float = image.astype(np.float32) / 255.0

        # 计算x和y方向的梯度
        grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # 归一化到0-255范围
        gradient = (gradient / (gradient.max() + 1e-8)) * 255

        return gradient.astype(np.uint8)
    except Exception as e:
        print(f"梯度计算失败: {e}")
        return np.zeros_like(image)

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

def calculate_depth_consistency_metrics(generated_depth, original_depth, generated_img=None, target_img=None):
    """
    计算生成图像深度图与原始深度图的一致性指标

    深度一致性指标说明：
    - Score_SSIM: 深度结构相似性分数，范围[0,1]，越接近1表示深度结构越相似
    - Score_MAE: 深度误差相似性分数，范围(0,1]，越接近1表示深度误差越小
    - Score_GradMAE: 深度梯度相似性分数，范围(0,1]，越接近1表示深度梯度越相似
    - Score_LPIPS: 感知相似性分数，范围(0,1]，越接近1表示生成图像越符合人眼观察
    - Composite_Score: 综合分数，范围[0,1]，越高表示深度一致性越好
      权重分配：Composite_Score = (0.35 * Score_SSIM) + (0.1 * Score_MAE) + (0.2 * Score_GradMAE) + (0.35 * Score_LPIPS)

    Args:
        generated_depth: 生成图像的深度图 (numpy array, HW, 0-255)
        original_depth: 原始深度图 (numpy array, HW, 0-255)
        generated_img: 生成图像 (numpy array, HWC, 0-255)，可选，用于计算Score_LPIPS
        target_img: 目标图像 (numpy array, HWC, 0-255)，可选，用于计算Score_LPIPS

    Returns:
        metrics: 包含各种深度一致性指标的字典
    """
    metrics = {}

    try:
        # 确保图像尺寸一致
        if generated_depth.shape != original_depth.shape:
            original_depth = cv2.resize(original_depth, (generated_depth.shape[1], generated_depth.shape[0]))

        # 转换为float类型进行计算
        gen_depth_float = generated_depth.astype(np.float32)
        orig_depth_float = original_depth.astype(np.float32)

        # 计算SSIM
        if SSIM_AVAILABLE:
            # 对于单通道深度图，使用multichannel=False
            ssim_score = ssim(gen_depth_float, orig_depth_float, data_range=255.0)
            score_ssim = float(ssim_score)  # Score_SSIM = SSIM(D, D_gen)
            metrics['score_ssim'] = score_ssim
        else:
            metrics['score_ssim'] = None

        # 计算MAE (平均绝对误差)
        mae = np.mean(np.abs(gen_depth_float - orig_depth_float))
        score_mae = 1.0 / (1.0 + mae)  # Score_MAE = 1 / (1 + MAE(D, D_gen))
        metrics['score_mae'] = float(score_mae)
        metrics['mae'] = float(mae)  # 保留原始MAE值用于参考

        # 计算梯度
        gen_gradient = calculate_gradient(generated_depth)
        orig_gradient = calculate_gradient(original_depth)

        # 计算梯度MAE
        grad_mae = np.mean(np.abs(gen_gradient.astype(np.float32) - orig_gradient.astype(np.float32)))
        score_grad_mae = 1.0 / (1.0 + grad_mae)  # Score_GradMAE = 1 / (1 + MAE(Grad(D), Grad(D_gen)))
        metrics['score_grad_mae'] = float(score_grad_mae)
        metrics['grad_mae'] = float(grad_mae)  # 保留原始梯度MAE值用于参考

        # 计算Score_LPIPS（基于生成图像和目标图像）
        score_lpips = 0.0
        if generated_img is not None and target_img is not None:
            try:
                # 使用现有的calculate_image_quality_metrics函数获取LPIPS值
                image_metrics = calculate_image_quality_metrics(generated_img, target_img)
                if image_metrics.get('lpips') is not None:
                    # Score_LPIPS = 1 / (1 + LPIPS_value)
                    score_lpips = 1.0 / (1.0 + image_metrics['lpips'])
                    metrics['lpips_value'] = float(image_metrics['lpips'])  # 保留原始LPIPS值
                else:
                    metrics['lpips_value'] = None
            except Exception as e:
                print(f"Warning: LPIPS calculation failed in depth consistency: {e}")
                metrics['lpips_value'] = None
        else:
            metrics['lpips_value'] = None

        metrics['score_lpips'] = float(score_lpips)

        # 计算综合分数 - 方案C：结构感知并重
        # Composite_Score = (0.35 × Score_SSIM) + (0.1 × Score_MAE) + (0.2 × Score_GradMAE) + (0.35 × Score_LPIPS)
        if metrics['score_ssim'] is not None:
            composite_score = (0.35 * metrics['score_ssim']) + (0.1 * metrics['score_mae']) + (0.2 * metrics['score_grad_mae']) + (0.35 * metrics['score_lpips'])
        else:
            # 如果SSIM不可用，重新分配权重：Score_MAE(0.154) + Score_GradMAE(0.308) + Score_LPIPS(0.538)
            # 权重按原比例缩放：原权重0.1+0.2+0.35=0.65，缩放因子为1/0.65≈1.538
            composite_score = (0.154 * metrics['score_mae']) + (0.308 * metrics['score_grad_mae']) + (0.538 * metrics['score_lpips'])

        metrics['composite_score'] = float(composite_score)

        # 添加便于理解的指标说明
        metrics['interpretation'] = {
            'score_ssim': "深度结构相似性，>0.65为较好",
            'score_mae': "深度误差相似性，>0.8为较好",
            'score_grad_mae': "深度梯度相似性，>0.8为较好",
            'score_lpips': "感知相似性，>0.7为较好",
            'composite_score': "综合深度一致性，>0.7为较好"
        }

    except Exception as e:
        print(f"Error calculating depth consistency metrics: {e}")
        # 返回默认值
        metrics = {
            'score_ssim': None,
            'score_mae': 0.0,
            'score_grad_mae': 0.0,
            'score_lpips': 0.0,
            'composite_score': 0.0,
            'mae': 255.0,
            'grad_mae': 255.0,
            'lpips_value': None,
            'interpretation': {
                'score_ssim': "深度结构相似性，>0.65为较好",
                'score_mae': "深度误差相似性，>0.8为较好",
                'score_grad_mae': "深度梯度相似性，>0.8为较好",
                'score_lpips': "感知相似性，>0.7为较好",
                'composite_score': "综合深度一致性，>0.7为较好"
            }
        }

    return metrics

def calculate_image_quality_metrics(generated_img, target_img):
    """
    计算生成图像与目标图像的质量指标（保留用于向后兼容）

    注意：这个函数已被calculate_depth_consistency_metrics取代，
    现在应该使用深度图一致性评估而不是图像相似性评估

    Args:
        generated_img: 生成的图像 (numpy array, HWC, 0-255)
        target_img: 目标图像 (numpy array, HWC, 0-255)

    Returns:
        metrics: 包含各种质量指标的字典
    """
    print("警告：使用了已弃用的calculate_image_quality_metrics函数")
    print("建议：使用calculate_depth_consistency_metrics进行深度一致性评估")

    metrics = {}

    try:
        # 确保图像尺寸一致
        if generated_img.shape != target_img.shape:
            target_img = cv2.resize(target_img, (generated_img.shape[1], generated_img.shape[0]))

        # 计算SSIM
        if SSIM_AVAILABLE:
            ssim_score = ssim(generated_img, target_img, multichannel=True, channel_axis=2, data_range=255)
            metrics['ssim'] = float(ssim_score)

        # 计算PSNR
        if SSIM_AVAILABLE:
            psnr_score = psnr(generated_img, target_img, data_range=255)
            metrics['psnr'] = float(psnr_score)

        # 计算LPIPS
        if LPIPS_AVAILABLE:
            try:
                if not hasattr(calculate_image_quality_metrics, 'lpips_model'):
                    calculate_image_quality_metrics.lpips_model = lpips.LPIPS(net='vgg').cuda()

                gen_tensor = torch.from_numpy(generated_img).float().cuda() / 255.0
                target_tensor = torch.from_numpy(target_img).float().cuda() / 255.0

                gen_tensor = gen_tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1

                with torch.no_grad():
                    lpips_score = calculate_image_quality_metrics.lpips_model(gen_tensor, target_tensor)
                metrics['lpips'] = float(lpips_score.item())
            except Exception as e:
                print(f"Warning: LPIPS calculation failed: {e}")
                metrics['lpips'] = None

        # 计算MSE
        mse = np.mean((generated_img.astype(float) - target_img.astype(float)) ** 2)
        metrics['mse'] = float(mse)

        # 计算MAE
        mae = np.mean(np.abs(generated_img.astype(float) - target_img.astype(float)))
        metrics['mae'] = float(mae)

    except Exception as e:
        print(f"Error calculating image quality metrics: {e}")
        metrics = {'ssim': None, 'psnr': None, 'lpips': None, 'mse': None, 'mae': None}

    return metrics

def calculate_composite_score(metrics):
    """
    计算综合评分，用于早停判断

    注意：此函数现在优先使用深度一致性指标，如果不可用则回退到图像质量指标

    深度一致性评分公式（方案C：结构感知并重）：
    Composite_Score = (0.35 * Score_SSIM) + (0.1 * Score_MAE) + (0.2 * Score_GradMAE) + (0.35 * Score_LPIPS)

    图像质量评分公式（回退选项，与深度一致性权重一致）：
    Composite_Score = (SSIM * 0.35) + (Score_MAE * 0.1) + ((1 - LPIPS) * 0.2) + (Score_LPIPS * 0.35)
    其中: Score_MAE = 1 / (1 + MAE), Score_LPIPS = 1 / (1 + LPIPS_value)

    Args:
        metrics: 包含各种质量指标的字典

    Returns:
        composite_score: 综合评分（越高越好）
    """
    # 优先使用深度一致性指标
    if 'composite_score' in metrics:
        # 如果是深度一致性指标，直接返回计算好的综合分数
        if 'score_ssim' in metrics:
            return metrics['composite_score']

    # 回退到原有的图像质量指标计算
    score = 0.0
    count = 0

    # 检查可用的指标
    has_ssim = metrics.get('ssim') is not None
    has_psnr = metrics.get('psnr') is not None
    has_lpips = metrics.get('lpips') is not None

    # 使用与深度一致性评分一致的权重分配（方案C：结构感知并重）
    # 对应公式: Composite_Score = (0.35 × Score_SSIM) + (0.1 × Score_MAE) + (0.2 × Score_GradMAE) + (0.35 × Score_LPIPS)
    ssim_weight = 0.35     # 对应Score_SSIM权重 (35%)
    mae_weight = 0.1       # 对应Score_MAE权重 (10%)
    grad_mae_weight = 0.2  # 对应Score_GradMAE权重 (20%)
    lpips_weight = 0.35    # 对应Score_LPIPS权重 (35%)

    # SSIM权重（0-1，越高越好） - 对应Score_SSIM
    if has_ssim:
        score += metrics['ssim'] * ssim_weight
        count += ssim_weight

    # MAE权重（0-∞，越低越好） - 对应Score_MAE
    if metrics.get('mae') is not None:
        # 使用与Score_MAE相同的转换公式: Score_MAE = 1 / (1 + MAE)
        mae_score = 1.0 / (1.0 + metrics['mae'])
        score += mae_score * mae_weight
        count += mae_weight

    # LPIPS权重（0-1，越低越好） - 对应Score_LPIPS
    if has_lpips:
        # 使用Score_LPIPS = 1 / (1 + LPIPS_value)转换公式
        lpips_score = 1.0 / (1.0 + metrics['lpips'])
        score += lpips_score * lpips_weight
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
        default=4,
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
    train_dataset = DepthDataset('laion_depth_training/laion_depth.txt')
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
    val_data = train_dataset[16]
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

    # 初始化深度估计器用于验证
    depth_estimator = initialize_depth_estimator("depth_weight/depth_anything_v2_vitl.pth")
    if depth_estimator is not None:
        device_used = "GPU" if torch.cuda.is_available() and next(depth_estimator.parameters()).is_cuda else "CPU"
        print(f"✅ 深度估计器已初始化（{device_used}模式），将用于深度一致性验证")
    else:
        print("⚠️ 深度估计器初始化失败，将回退到图像相似性验证")

    # optimizer
    params = list(model_ad.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    # 使用ReduceLROnPlateau调度器，当composite_score不再提升时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # 最大化composite_score
        factor=0.5,           # 学习率衰减因子，每次降为原来的0.5
        patience=5,           # 早停耐心值的1/5，连续5次无改善则降低学习率
        min_lr=1e-8,          # 最小学习率阈值
        threshold=1e-6,       # 判断改善的最小阈值
        threshold_mode='abs'  # 绝对阈值模式
    )

    # 重新启用早停机制，基于图像质量评估
    best_val_score = -float('inf')  # 越大越好（SSIM、PSNR）
    worst_lpips = float('inf')      # 越小越好（LPIPS）
    patience = 20  # 早停耐心值
    patience_counter = 0  # 早停计数器
    best_model_state = None  # 最佳模型状态

    # 早停计算说明:
    # - 数据集: 12000张图片, 批量大小: 6, 每epoch: 2000次迭代
    # - 验证频率: 每2000次迭代 (每epoch验证一次)
    # - 早停条件: 连续15次验证无改善 = 连续15个epoch无改善



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
    logger.info(f"  Batch size per device = {opt.bsize}")
    logger.info(f"  Total optimization steps = {num_update_steps_per_epoch * opt.epochs}")
    logger.info(f"  Print frequency = {opt.print_fq}")
    logger.info(f"  Starting current_iter = {current_iter}")
    logger.info(f"  Initial learning rate = {optimizer.param_groups[0]['lr']:.2e}")
    logger.info(f"  Learning rate scheduler = ReduceLROnPlateau (patience=5, factor=0.1, min_lr=1e-8)")
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

    # 训练循环，支持早停和梯度累积
    early_stop = False
    optimizer.zero_grad()  # 在开始时清零梯度

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

            current_iter += 1  # 递增迭代计数器

            if current_iter%opt.print_fq == 0:
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
            if current_iter%config['training']['save_freq'] == 0:
                save_filename = f'model_ad_{current_iter}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                state_dict = model_ad.state_dict()
                save_dict = {}
                for key, param in state_dict.items():
                    save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)

                # save state
                state = {'epoch': epoch,
                         'iter': current_iter,
                         'optimizers': optimizer.state_dict()
                         }
                save_filename = f'{current_iter}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)

                # 记录检查点保存信息到日志
                logger.info(f"Saved checkpoint at epoch {epoch}, iter {current_iter}")

            # val
            if current_iter% opt.val_iter == 0:  # Always run validation for single GPU training
                gen_image_count+=1

                val_data = train_dataset[16]

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

                    # 获取原始深度图用于比较
                    original_depth = tensor2img(val_data['depth'])  # 获取原始深度图
                    original_depth = cv2.cvtColor(original_depth, cv2.COLOR_BGR2GRAY)  # 转为单通道

                    for id_sample, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255.*x_sample
                        generated_img = x_sample.astype(np.uint8)

                        # 从生成图像中提取深度图
                        if depth_estimator is not None:
                            # 新的验证方式：比较生成图像的深度图与原始深度图
                            generated_depth = extract_depth_from_image(depth_estimator, generated_img)

                            # 获取目标图像用于Score_LPIPS计算
                            target_img = tensor2img(val_data['im'])

                            # 计算深度一致性指标（包含Score_LPIPS）
                            metrics = calculate_depth_consistency_metrics(generated_depth, original_depth, generated_img, target_img)

                            # 保存深度图用于可视化
                            depth_vis_path = os.path.join(experiments_root, 'visualization',
                                                          f'depth_e{epoch:04d}_s{gen_image_count:04d}_sample{id_sample}.png')
                            cv2.imwrite(depth_vis_path, generated_depth)

                            # 如果启用wandb，记录深度图对比
                            if opt.use_wandb and wandb_available:
                                wandb_depth_comparison = wandb.Image(
                                    generated_depth,
                                    caption=f"Depth from Generated Image - Epoch {epoch}, Sample {gen_image_count}-{id_sample}"
                                )
                                wandb.log({
                                    f"val/depth_e{epoch:04d}_s{gen_image_count:04d}_{id_sample}": wandb_depth_comparison
                                }, step=current_iter)
                        else:
                            # 回退到旧的验证方式：比较生成图像与目标图像
                            target_img = tensor2img(val_data['im'])  # 获取目标图像
                            metrics = calculate_image_quality_metrics(generated_img, target_img)

                        # 记录指标到日志
                        # 过滤掉interpretation字典，只显示数值指标
                        display_metrics = {k: v for k, v in metrics.items() if k != 'interpretation'}
                        metrics_str = ", ".join([f"{k}: {v:.4f}" if v is not None else f"{k}: N/A"
                                                for k, v in display_metrics.items()])
                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(f"=== Validation at iteration {current_iter} (第{gen_image_count}次验证) ===")
                        logger.info(f"Current learning rate: {current_lr:.2e}")

                        if depth_estimator is not None:
                            logger.info(f"Depth consistency metrics - {metrics_str}")
                            if 'interpretation' in metrics:
                                logger.info("深度一致性指标解读:")
                                for metric, interpretation in metrics['interpretation'].items():
                                    logger.info(f"  {metric}: {interpretation}")
                        else:
                            logger.info(f"Image similarity metrics (回退模式) - {metrics_str}")
                            logger.info("指标解读: SSIM>0.5较好, PSNR>30dB较好, LPIPS<0.2较好, MSE/MAE越低越好")

                        # 计算综合评分
                        composite_score = calculate_composite_score(metrics)

                        # 记录到wandb - 包含所有验证指标
                        if opt.use_wandb and wandb_available:
                            wandb_metrics = {}
                            # 记录各个指标，过滤掉interpretation
                            for k, v in metrics.items():
                                if v is not None and k != 'interpretation':
                                    if depth_estimator is not None:
                                        # 深度一致性指标
                                        wandb_metrics[f"val/depth_{k}"] = v
                                    else:
                                        # 图像相似性指标
                                        wandb_metrics[f"val/{k}"] = v

                            # 记录综合评分
                            if depth_estimator is not None:
                                wandb_metrics["val/depth_composite_score"] = composite_score
                                wandb_metrics["val/depth_best_score"] = best_val_score
                            else:
                                wandb_metrics["val/composite_score"] = composite_score
                                wandb_metrics["val/best_score"] = best_val_score

                            # 记录patience信息
                            wandb_metrics["val/patience_counter"] = patience_counter
                            # 记录当前学习率
                            current_lr = optimizer.param_groups[0]['lr']
                            wandb_metrics["train/learning_rate"] = current_lr

                            # 记录验证模式
                            wandb_metrics["val/validation_mode"] = "depth_consistency" if depth_estimator is not None else "image_similarity"

                            wandb.log(wandb_metrics, step=current_iter)

                        # 早停判断和最佳模型保存
                        # 早停机制：每次验证时判断，如果连续15次验证没有改善则停止训练
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

                            validation_type = "深度一致性" if depth_estimator is not None else "图像相似性"
                            logger.info(f"🎉 NEW BEST MODEL! iteration {current_iter} - {validation_type}评测得分: {composite_score:.4f}")
                            logger.info(f"✅ 最佳模型已保存到: model_ad_best.pth")
                        else:
                            # 性能没有改善，增加早停计数器
                            patience_counter += 1
                            validation_type = "深度一致性" if depth_estimator is not None else "图像相似性"
                            logger.info(f"❌ Score not improved - iteration {current_iter}: {validation_type}评测得分 {composite_score:.4f}, Best: {best_val_score:.4f}")
                            logger.info(f"Validation score did not improve. Patience counter: {patience_counter}/{patience}")
                            logger.info(f"早停说明: 连续{patience}次验证无改善将自动停止训练，当前第{patience_counter}次")

                        # 检查是否需要早停
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered after {patience} validations without improvement")
                            logger.info(f"Best composite score: {best_val_score:.4f}")
                            logger.info(f"训练已早停: 在连续{patience}次验证（约{patience * opt.val_iter}次迭代）中性能未提升")
                            early_stop = True
                            break

                        # 学习率调度：使用ReduceLROnPlateau
                        old_lr = optimizer.param_groups[0]['lr']
                        scheduler.step(composite_score)  # 根据composite_score调整学习率
                        new_lr = optimizer.param_groups[0]['lr']

                        # 如果学习率发生变化，记录相关信息
                        if new_lr != old_lr:
                            logger.info(f"🔽 学习率调整: {old_lr:.2e} → {new_lr:.2e} (iteration {current_iter})")
                            if opt.use_wandb and wandb_available:
                                wandb_metrics["train/learning_rate"] = new_lr

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

                    logger.info(f"Validation completed at epoch {epoch}, iter {current_iter}")


            # 正常训练流程
        # 检查是否早停，如果早停则跳出epoch循环
        if early_stop:
            break

        # 记录当前epoch结束时的学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch} finished. Current learning rate: {current_lr:.2e}')

    # 关闭进度条
    progress_bar.close()

    # 如果早停被触发，恢复最佳模型状态
    if early_stop and best_model_state is not None:
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