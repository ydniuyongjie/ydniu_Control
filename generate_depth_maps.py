#!/usr/bin/env python3
"""
使用Depth Anything V2为LAION数据集生成深度图
专门为T2I-Adapter训练设计
"""

import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import webdataset as wds

# 添加Depth-Anything-V2到Python路径
import sys
import os
depth_anything_path = "/home/tniuyj/code/T2I-Adapter/Depth-Anything-V2"
if depth_anything_path not in sys.path:
    sys.path.insert(0, depth_anything_path)

# 检查Depth Anything V2是否可用
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_V2_AVAILABLE = True
    print("✅ Depth Anything V2 可用")
except ImportError as e:
    print(f"❌ 无法导入Depth Anything V2: {e}")
    print("请确保Depth-Anything-V2已正确安装")
    DEPTH_ANYTHING_V2_AVAILABLE = False
    exit(1)

class DepthGenerator:
    def __init__(self, weight_path="depth_weight/depth_anything_v2_vitl.pth", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载Depth Anything V2模型
        print("加载Depth Anything V2模型...")
        self.model = DepthAnythingV2(
            encoder='vitl',  # ViT-Large
            features=256,
            out_channels=[256, 512, 1024, 1024]
        )

        # 加载权重
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载成功: {weight_path}")

    def generate_depth(self, image):
        """生成单张图片的深度图"""
        # 确保图像是BGR格式的numpy数组
        if isinstance(image, str):
            image = cv2.imread(image)
        elif hasattr(image, 'save'):  # PIL Image
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if image is None:
            raise ValueError("无法读取图像")

        # 确保图像数据类型正确
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        # 使用Depth Anything V2生成深度图
        with torch.no_grad():
            depth = self.model.infer_image(image)  # HxW raw depth map

        # 归一化到0-255范围并转换为uint8
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)

        return depth

def create_dataset_list(output_path, processed_count):
    """创建数据集列表文件laion_depth.txt"""
    if processed_count == 0:
        print("没有处理的样本，跳过创建数据集列表")
        return

    dataset_file = output_path / "laion_depth.txt"

    try:
        # 获取所有png文件并按名称排序
        png_files = sorted(output_path.glob("*.png"))

        # 过滤出原始图像文件（排除深度图）
        image_files = [f for f in png_files if not f.name.endswith(".depth.png")]

        print(f"找到 {len(image_files)} 个原始图像文件")

        # 写入数据集列表文件
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for img_file in image_files:
                f.write(f"{img_file.name}\n")

        print(f"✅ 数据集列表文件已创建: {dataset_file}")
        print(f"包含 {len(image_files)} 个图像文件")

    except Exception as e:
        print(f"❌ 创建数据集列表文件时出错: {e}")

def process_webdataset(input_dir, output_dir, weight_path):
    """处理webdataset格式的数据"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有tar文件
    tar_files = list(input_path.glob("*.tar"))
    if not tar_files:
        print(f"在 {input_dir} 中没有找到tar文件")
        return

    print(f"找到 {len(tar_files)} 个tar文件")

    # 初始化深度生成器
    depth_generator = DepthGenerator(weight_path, "cuda")

    processed_count = 0
    error_count = 0

    # 处理每个tar文件
    for tar_file in tqdm(tar_files, desc="处理tar文件"):
        try:
            # 创建webdataset数据加载器
            dataset = wds.WebDataset(str(tar_file)).decode("pil")

            for sample in dataset:
                try:
                    # 检查样本结构
                    if not isinstance(sample, dict):
                        continue

                    # 获取图像和文本，支持jpg和png格式
                    image = None
                    caption_text = None

                    # 检查是否有jpg或png格式的图像
                    if 'jpg' in sample:
                        image = sample['jpg']
                    elif 'png' in sample:
                        image = sample['png']

                    # 检查是否有对应的文本文件
                    if image is not None and 'txt' in sample:
                        caption_text = sample['txt']

                        # 处理caption
                        if isinstance(caption_text, bytes):
                            caption = caption_text.decode('utf-8').strip()
                        else:
                            caption = str(caption_text).strip()

                        # 获取文件名
                        filename = sample.get('__key__', f"image_{processed_count:06d}")
                        if isinstance(filename, bytes):
                            filename = filename.decode('utf-8')
                        else:
                            filename = str(filename)

                        # 清理文件名（移除扩展名）
                        base_name = filename.replace('.jpg', '').replace('.png', '')

                        # 生成深度图
                        depth = depth_generator.generate_depth(image)

                        # 保存文件（统一保存为png格式）
                        image_path = output_path / f"{base_name}.png"
                        depth_path = output_path / f"{base_name}.depth.png"
                        txt_path = output_path / f"{base_name}.txt"

                        # 保存图像
                        if not image_path.exists():
                            image_array = np.array(image)
                            if hasattr(image, 'mode') and image.mode == 'RGB':
                                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(image_path), image_array)

                        # 保存深度图
                        cv2.imwrite(str(depth_path), depth)

                        # 保存文本
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(caption if caption else "A high quality photograph")

                        processed_count += 1

                        if processed_count % 1 == 0:  # 每处理一个样本都打印
                            print(f"已处理 {processed_count} 个样本 - {base_name}")

                except Exception as e:
                    error_count += 1
                    print(f"处理样本时出错: {e}")
                    continue

        except Exception as e:
            print(f"处理tar文件 {tar_file} 时出错: {e}")
            continue

    print(f"\n处理完成!")
    print(f"成功处理: {processed_count}")
    print(f"失败数量: {error_count}")
    print(f"输出目录: {output_path}")
    print(f"文件格式: {{filename}}.png, {{filename}}.depth.png, {{filename}}.txt")

    # 创建数据集列表文件
    create_dataset_list(output_path, processed_count)

def main():
    parser = argparse.ArgumentParser(description="使用Depth Anything V2为LAION数据集生成深度图")
    parser.add_argument("--input_dir", type=str, default="laion_aesthetics_v2__images",
                        help="webdataset数据目录")
    parser.add_argument("--output_dir", type=str, default="laion_depth_training",
                        help="输出目录")
    parser.add_argument("--weight_path", type=str, default="depth_weight/depth_anything_v2_vitl.pth",
                        help="Depth Anything V2权重文件路径")

    args = parser.parse_args()

    print("LAION数据集深度图生成器 (Depth Anything V2)")
    print("=" * 50)

    # 检查权重文件
    if not os.path.exists(args.weight_path):
        print(f"权重文件不存在: {args.weight_path}")
        print("请确保depth_anything_v2_vitl.pth文件存在")
        return

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        return

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"权重文件: {args.weight_path}")
    print("=" * 50)

    # 开始处理
    process_webdataset(args.input_dir, args.output_dir, args.weight_path)

if __name__ == "__main__":
    main()