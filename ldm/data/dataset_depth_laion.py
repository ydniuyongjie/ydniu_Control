import json
import cv2
import os
from basicsr.utils import img2tensor


class LaionDepthDataset():
    def __init__(self, meta_file):
        super(LaionDepthDataset, self).__init__()

        self.files = []
        # 获取meta_file所在的目录
        self.base_dir = os.path.dirname(os.path.abspath(meta_file))

        with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.strip()
                if img_name:  # 跳过空行
                    # 构建完整路径
                    full_img_path = os.path.join(self.base_dir, f"{img_name}")
                    depth_img_path = os.path.join(self.base_dir, f"{img_name.rsplit('.', 1)[0]}.depth.png")
                    txt_path = os.path.join(self.base_dir, f"{img_name.rsplit('.', 1)[0]}.txt")

                    # 检查文件是否存在
                    if os.path.exists(full_img_path) and os.path.exists(depth_img_path):
                        self.files.append({
                            'img_path': full_img_path,
                            'depth_img_path': depth_img_path,
                            'txt_path': txt_path
                        })

        print(f"Loaded {len(self.files)} samples from {meta_file}")

    def __getitem__(self, idx):
        file = self.files[idx]

        # 读取原始图像
        if not os.path.exists(file['img_path']):
            raise FileNotFoundError(f"原始图像文件不存在: {file['img_path']}")
        im = cv2.imread(file['img_path'])
        if im is None:
            raise ValueError(f"无法读取原始图像文件: {file['img_path']}")

        # 调整图像大小为512*512
        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR)
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        # 读取深度图像
        if not os.path.exists(file['depth_img_path']):
            raise FileNotFoundError(f"深度图像文件不存在: {file['depth_img_path']}")
        depth = cv2.imread(file['depth_img_path'])
        if depth is None:
            raise ValueError(f"无法读取深度图像文件: {file['depth_img_path']}")

        # 调整深度图像大小为512*512
        depth = cv2.resize(depth, (512, 512), interpolation=cv2.INTER_LINEAR)
        depth = img2tensor(depth, bgr2rgb=True, float32=True) / 255.

        # 读取文本描述
        if not os.path.exists(file['txt_path']):
            # 如果没有文本文件，使用默认描述
            sentence = "A high quality photograph"
        else:
            try:
                with open(file['txt_path'], 'r', encoding='utf-8') as f:
                    sentence = f.read().strip()
                if not sentence:
                    sentence = "A high quality photograph"
            except Exception:
                sentence = "A high quality photograph"

        return {'im': im, 'depth': depth, 'sentence': sentence}

    def __len__(self):
        return len(self.files)