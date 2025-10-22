#!/usr/bin/env python3
"""
为下载的图像生成高质量的深度图
使用MiDaS等深度估计模型
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
import webdataset as wds
import tarfile
import tempfile
import shutil

# 尝试导入MiDaS模型
try:
    from midas.dpt_depth import DPTDepthModel
    from midas.midas_net import MidasNet
    from midas.midas_net_custom import MidasNet_small
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    MIDAS_AVAILABLE = True
except ImportError:
    print("MiDaS not available. Using simple depth estimation.")
    MIDAS_AVAILABLE = False

class DepthEstimator:
    def __init__(self, model_type="dpt_large", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.model_type = model_type

        if MIDAS_AVAILABLE:
            self.load_midas_model()
        else:
            self.setup_simple_model()

    def load_midas_model(self):
        """加载MiDaS深度估计模型"""
        try:
            print(f"Loading MiDaS model: {self.model_type}")

            # 模型路径映射
            model_paths = {
                "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
                "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
                "midas_v21": "weights/midas_v21-f6b98070.pt",
                "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt"
            }

            if self.model_type in model_paths:
                model_path = model_paths[self.model_type]
                if not os.path.exists(model_path):
                    print(f"Downloading {self.model_type} model...")
                    self.download_midas_model(model_type, model_path)

            # 加载模型
            if "dpt" in self.model_type:
                self.model = DPTDepthModel(
                    path=model_path,
                    backbone="vitl16_384" if self.model_type == "dpt_large" else "vitb_rn384"
                )
            elif "small" in self.model_type:
                self.model = MidasNet_small(model_path, non_negative=True)
            else:
                self.model = MidasNet(model_path, non_negative=True)

            self.model.to(self.device)
            self.model.eval()

            # 设置变换
            if "dpt" in self.model_type:
                self.transform = transforms.Compose([
                    Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        method="cv2",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ])
            else:
                self.transform = transforms.Compose([
                    Resize(384, 384),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ])

            print(f"✅ MiDaS model loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load MiDaS: {e}")
            self.setup_simple_model()

    def download_midas_model(self, model_type, model_path):
        """下载MiDaS模型"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        url_map = {
            "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
            "midas_v21": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21-f6b98070.pt",
            "midas_v21_small": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt"
        }

        if model_type in url_map:
            import urllib.request
            urllib.request.urlretrieve(url_map[model_type], model_path)

    def setup_simple_model(self):
        """设置简单的深度估计模型"""
        print("Using simple depth estimation method")
        self.model = None
        self.transform = None

    def estimate_depth(self, image):
        """估计图像深度"""
        if self.model is not None:
            return self.estimate_depth_midas(image)
        else:
            return self.estimate_depth_simple(image)

    def estimate_depth_midas(self, image):
        """使用MiDaS估计深度"""
        try:
            # 转换为PIL图像
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    image = Image.fromarray(image)

            # 应用变换
            input_batch = self.transform({"image": image})["image"]
            input_batch = input_batch.unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                prediction = self.model(input_batch)

                # 后处理
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # 转换为numpy
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min())  # 归一化到[0,1]
            depth = (depth * 255).astype(np.uint8)

            return depth

        except Exception as e:
            print(f"MiDaS depth estimation failed: {e}")
            return self.estimate_depth_simple(image)

    def estimate_depth_simple(self, image):
        """简单的深度估计方法"""
        try:
            if isinstance(image, str):
                image = cv2.imread(image)
            elif isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            if image is None:
                raise ValueError("Cannot load image")

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用多种方法组合生成深度
            # 1. 亮度梯度
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # 2. 距离变换（边缘远离的区域更远）
            edges = cv2.Canny(gray, 50, 150)
            distance_transform = cv2.distanceTransform(255-edges, cv2.DIST_L2, 5)

            # 3. 亮度信息
            brightness = gray.copy()

            # 组合深度信息
            # 归一化各个组件
            gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)
            distance_transform = (distance_transform - distance_transform.min()) / (distance_transform.max() - distance_transform.min() + 1e-8)
            brightness = (255 - brightness) / 255.0  # 反转亮度

            # 加权组合
            depth = (0.3 * gradient_magnitude + 0.4 * distance_transform + 0.3 * brightness)
            depth = (depth * 255).astype(np.uint8)

            # 应用高斯滤波平滑
            depth = cv2.GaussianBlur(depth, (5, 5), 0)

            return depth

        except Exception as e:
            print(f"Simple depth estimation failed: {e}")
            # 最后的备用方案
            return np.random.randint(0, 256, (512, 512), dtype=np.uint8)

def generate_depth_for_webdataset(webdataset_dir, output_dir, model_type="dpt_large"):
    """为webdataset格式的数据生成深度图"""
    webdataset_path = Path(webdataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有tar文件
    tar_files = list(webdataset_path.glob("*.tar"))
    if not tar_files:
        print(f"❌ 在目录 {webdataset_dir} 中没有找到tar文件")
        return

    print(f"📦 找到 {len(tar_files)} 个tar文件")

    # 初始化深度估计器
    estimator = DepthEstimator(model_type=model_type)

    # 创建输出目录结构 - 适合T2I-Adapter训练
    # 所有文件放在同一目录下，便于训练使用
    output_path.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    success_count = 0
    error_count = 0

    # 处理每个tar文件
    for tar_file in tqdm(tar_files, desc="Processing tar files"):
        try:
            # 创建webdataset数据加载器
            dataset = wds.WebDataset(str(tar_file)).decode("pil")

            for sample in dataset:
                try:
                    # 获取图像、txt和json
                    if 'jpg' in sample and 'txt' in sample:
                        image = sample['jpg']
                        # 从txt文件读取真正的caption
                        caption_text = sample['txt']
                        if isinstance(caption_text, bytes):
                            caption = caption_text.decode('utf-8').strip()
                        else:
                            caption = str(caption_text).strip()

                        # 从webdataset样本中获取原始文件名
                        original_filename = None
                        if '__key__' in sample:
                            # 使用webdataset的key作为文件名
                            original_filename = sample['__key__']
                        else:
                            # 从jpg文件名中提取
                            jpg_data = sample.get('jpg', b'')
                            # 尝试从其他方式获取文件名，使用计数器作为备选
                            original_filename = f"image_{total_processed:06d}"

                        # 生成文件名
                        image_filename = f"{original_filename}.jpg"
                        depth_filename = f"{original_filename}.depth.png"

                        # 转换PIL图像到numpy
                        image_array = np.array(image)
                        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                            # RGB到BGR转换
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                            # RGBA到BGR转换
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

                        # 生成深度图
                        depth = estimator.estimate_depth(image_array)

                        # 保存原图和深度图到同一目录
                        image_save_path = output_path / image_filename
                        depth_save_path = output_path / depth_filename
                        txt_save_path = output_path / f"{original_filename}.txt"

                        cv2.imwrite(str(image_save_path), image_array)
                        cv2.imwrite(str(depth_save_path), depth)

                        try:
                            # 保存真正的文本描述
                            with open(txt_save_path, 'w', encoding='utf-8') as f:
                                if caption and len(caption.strip()) > 0:
                                    f.write(caption.strip())
                                else:
                                    f.write("A high quality photograph")  # 备用描述

                            success_count += 1
                            total_processed += 1

                        except Exception as e:
                            error_count += 1
                            print(f"保存文件时出错: {e}")
                            continue

                except Exception as e:
                    error_count += 1
                    print(f"解析样本时出错: {e}")
                    continue

        except Exception as e:
            print(f"处理tar文件 {tar_file} 时出错: {e}")
            continue

    # 生成数据集列表文件 - 适合T2I-Adapter训练
    dataset_list_file = output_path / "dataset_list.txt"
    processed_files = []

    # 收集所有成功处理的文件
    for file_path in output_path.glob("*.jpg"):
        if (output_path / f"{file_path.stem}.depth.png").exists():
            processed_files.append(file_path.name)

    with open(dataset_list_file, 'w', encoding='utf-8') as f:
        for filename in processed_files:
            f.write(f"{filename}\n")

    print(f"\n✅ 处理完成!")
    print(f"总共处理: {total_processed}")
    print(f"成功生成: {success_count}")
    print(f"失败数量: {error_count}")
    print(f"成功率: {success_count/(total_processed+1e-8)*100:.2f}%")
    print(f"数据集保存在: {output_path}")
    print(f"数据集列表: {dataset_list_file}")
    print(f"📋 可用于训练的数据集格式:")
    print(f"   - 原图: {output_path}/{{filename}}.jpg")
    print(f"   - 深度图: {output_path}/{{filename}}.depth.png")
    print(f"   - 文本: {output_path}/{{filename}}.txt")
    print(f"   - 列表: {dataset_list_file}")

def generate_depth_for_dataset(images_dir, output_dir, model_type="dpt_large"):
    """为数据集生成深度图"""
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_path.glob(ext))

    print(f"Found {len(image_files)} images")

    # 初始化深度估计器
    estimator = DepthEstimator(model_type=model_type)

    # 生成进度条
    progress_bar = tqdm(image_files, desc="Generating depth maps")

    success_count = 0
    error_count = 0

    for image_file in progress_bar:
        try:
            # 生成深度图
            depth = estimator.estimate_depth(str(image_file))

            # 保存深度图
            depth_filename = output_path / f"{image_file.stem}.depth.png"
            cv2.imwrite(str(depth_filename), depth)

            # 生成文本描述（如果不存在）
            txt_file = images_path / f"{image_file.stem}.txt"
            output_txt_file = output_path / f"{image_file.stem}.txt"

            if not output_txt_file.exists() and txt_file.exists():
                import shutil
                shutil.copy2(txt_file, output_txt_file)
            elif not output_txt_file.exists():
                # 创建默认描述
                with open(output_txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"A high quality photograph with depth information")

            success_count += 1

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            error_count += 1

    # 创建数据集列表文件 - 适合T2I-Adapter训练
    dataset_file = output_path / "dataset_list.txt"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for image_file in image_files:
            if (output_path / f"{image_file.stem}.depth.png").exists():
                f.write(f"{image_file.name}\n")

    print(f"\n✅ Depth generation complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Depth maps saved to: {output_path}")
    print(f"Dataset list: {dataset_file}")
    print(f"📋 可用于训练的数据集格式:")
    print(f"   - 原图: {output_path}/{{filename}}.{{ext}}")
    print(f"   - 深度图: {output_path}/{{filename}}.depth.png")
    print(f"   - 文本: {output_path}/{{filename}}.txt")
    print(f"   - 列表: {dataset_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for downloaded images")

    # 创建互斥组：要么使用images_dir，要么使用webdataset_dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images_dir", type=str,
                        help="Directory containing downloaded images")
    group.add_argument("--webdataset_dir", type=str,
                        help="Directory containing webdataset tar files")

    parser.add_argument("--output_dir", type=str, default="laion_with_depth",
                        help="Output directory for images with depth maps")
    parser.add_argument("--model_type", type=str, default="dpt_large",
                        choices=["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21_small", "simple"],
                        help="Depth estimation model type")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    print("🔬 Depth Map Generator")
    if args.images_dir:
        print(f"Images directory: {args.images_dir}")
        input_type = "images"
    else:
        print(f"Webdataset directory: {args.webdataset_dir}")
        input_type = "webdataset"
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_type}")
    print(f"Device: {args.device}")

    # 检查webdataset依赖
    if input_type == "webdataset":
        try:
            import webdataset
        except ImportError:
            print("❌ webdataset未安装，请安装: pip install webdataset")
            return

    # 如果使用MiDaS，安装依赖
    if MIDAS_AVAILABLE and args.model_type != "simple":
        try:
            # 尝试安装MiDaS
            import subprocess
            subprocess.check_call(["pip", "install", "transformers", "timm"])
        except:
            print("Warning: Could not install additional dependencies. Using simple depth estimation.")

    # 根据输入类型选择处理函数
    if input_type == "webdataset":
        generate_depth_for_webdataset(args.webdataset_dir, args.output_dir, args.model_type)
    else:
        generate_depth_for_dataset(args.images_dir, args.output_dir, args.model_type)

if __name__ == "__main__":
    main()