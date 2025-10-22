import os
from datasets import load_dataset
from PIL import Image
import numpy as np

def save_dataset_items():
    # 加载数据集
    dataset = load_dataset("Nahrawy/VIDIT-Depth-ControlNet")
    
    # 创建输出目录
    output_dir = "depth"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建用于存储文件名的列表
    image_names = []
    
    # 处理每个样本
    for idx, item in enumerate(dataset['train']):
        # 生成6位数字的文件名
        file_name = f"{idx:06d}"
        
        # 保存原始图像
        image = item['image']
        image.save(os.path.join(output_dir, f"{file_name}.png"))
        
        # 保存depth map
        depth_map = item['depth_map']
        depth_map.save(os.path.join(output_dir, f"{file_name}.depth.png"))
        
        # 保存caption
        with open(os.path.join(output_dir, f"{file_name}.txt"), 'w') as f:
            f.write(item['caption'])
        
        # 记录图像文件名
        image_names.append(f"{file_name}.png")
    
    # 保存所有图像文件名到depth.txt
    with open(os.path.join(output_dir, "depth.txt"), 'w') as f:
        f.write('\n'.join(image_names))

if __name__ == "__main__":
    save_dataset_items()
