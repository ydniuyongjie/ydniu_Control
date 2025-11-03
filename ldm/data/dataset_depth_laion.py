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


class LaionDepthDatasetWithValidation():
    """
    带验证功能的LAION深度数据集类
    支持训练集和验证集的划分
    """
    def __init__(self, meta_file, validation_split=0.1, seed=42, is_train=True):
        super(LaionDepthDatasetWithValidation, self).__init__()

        self.files = []
        self.base_dir = os.path.dirname(os.path.abspath(meta_file))

        # 读取所有文件
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.strip()
                if img_name:
                    full_img_path = os.path.join(self.base_dir, f"{img_name}")
                    depth_img_path = os.path.join(self.base_dir, f"{img_name.rsplit('.', 1)[0]}.depth.png")
                    txt_path = os.path.join(self.base_dir, f"{img_name.rsplit('.', 1)[0]}.txt")

                    if os.path.exists(full_img_path) and os.path.exists(depth_img_path):
                        self.files.append({
                            'img_path': full_img_path,
                            'depth_img_path': depth_img_path,
                            'txt_path': txt_path
                        })

        # 划分训练集和验证集
        total_samples = len(self.files)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size

        # 设置随机种子以确保划分的一致性
        import random
        random.seed(seed)
        random.shuffle(self.files)

        if is_train:
            self.files = self.files[:train_size]
            print(f"训练集: {len(self.files)} 个样本")
        else:
            self.files = self.files[train_size:]
            print(f"验证集: {len(self.files)} 个样本")

    def __getitem__(self, idx):
        return LaionDepthDataset.__getitem__(self, idx)

    def __len__(self):
        return len(self.files)


def create_dataset_from_directory(data_dir):
    """
    从数据目录直接创建数据集（不使用meta文件）
    适用于测试或小规模数据集
    """
    import glob

    base_dir = os.path.abspath(data_dir)
    png_files = glob.glob(os.path.join(base_dir, "*.png"))

    # 过滤掉深度图文件
    image_files = [f for f in png_files if not f.endswith(".depth.png")]

    print(f"在目录 {data_dir} 中找到 {len(image_files)} 个图像文件")

    dataset = LaionDepthDataset.__new__(LaionDepthDataset)
    dataset.base_dir = base_dir
    dataset.files = []

    for img_path in sorted(image_files):
        depth_img_path = img_path.rsplit('.', 1)[0] + '.depth.png'
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'

        # 只包含所有文件都存在的样本
        if os.path.exists(depth_img_path) and os.path.exists(txt_path):
            dataset.files.append({
                'img_path': img_path,
                'depth_img_path': depth_img_path,
                'txt_path': txt_path
            })

    print(f"有效样本数量: {len(dataset.files)}")
    return dataset


if __name__ == "__main__":
    # 测试数据集加载（不依赖basicsr）
    meta_file = "laion_depth_training/laion_depth.txt"

    if os.path.exists(meta_file):
        print("=== 测试LaionDepthDataset文件路径解析 ===")
        try:
            # 只测试文件路径解析，不测试数据加载
            import os
            base_dir = os.path.dirname(os.path.abspath(meta_file))
            files = []

            with open(meta_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    img_name = line.strip()
                    if img_name:
                        full_img_path = os.path.join(base_dir, f"{img_name}")
                        depth_img_path = os.path.join(base_dir, f"{img_name.rsplit('.', 1)[0]}.depth.png")
                        txt_path = os.path.join(base_dir, f"{img_name.rsplit('.', 1)[0]}.txt")

                        if os.path.exists(full_img_path) and os.path.exists(depth_img_path):
                            files.append({
                                'img_path': full_img_path,
                                'depth_img_path': depth_img_path,
                                'txt_path': txt_path
                            })

            print(f"数据集大小: {len(files)}")

            if len(files) > 0:
                # 显示前几个文件路径
                print("前5个样本的文件路径:")
                for i, file in enumerate(files[:5]):
                    print(f"  {i+1}. 图像: {os.path.basename(file['img_path'])}")
                    print(f"     深度图: {os.path.basename(file['depth_img_path'])}")
                    print(f"     文本: {os.path.basename(file['txt_path'])}")

                # 验证文件格式
                first_file = files[0]
                img_name = first_file['img_path']
                depth_name = first_file['depth_img_path']
                txt_name = first_file['txt_path']

                print(f"\n文件格式验证:")
                print(f"✅ 图像文件存在: {os.path.exists(img_name)}")
                print(f"✅ 深度图文件存在: {os.path.exists(depth_name)}")
                print(f"✅ 文本文件存在: {os.path.exists(txt_name)}")

                # 读取文本内容验证
                with open(txt_name, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    print(f"✅ 文本内容示例: {content}")

                print("✅ 数据集路径解析成功!")
            else:
                print("❌ 数据集为空")

        except Exception as e:
            print(f"❌ 数据集测试失败: {e}")
    else:
        print(f"❌ 元数据文件不存在: {meta_file}")