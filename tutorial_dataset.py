import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from basicsr.utils import img2tensor


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # 以灰度模式读取source图像，得到单通道图像
        source = cv2.imread('./fill50k/' + source_filename, cv2.IMREAD_GRAYSCALE)
        source = cv2.resize(source, (512, 512))
        # 添加通道维度，然后转换为tensor
        source = source[:, :, None]  # 添加通道维度 (H, W) -> (H, W, 1)
        source = img2tensor(source, bgr2rgb=False, float32=True) / 255.
        
        target = cv2.imread('./fill50k/' + target_filename)
        target = cv2.resize(target, (512, 512))
        target = img2tensor(target, bgr2rgb=True, float32=True) / 255.

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # # Normalize target images to [-1, 1].
        # target = (target.astype(np.float32) / 127.5) - 1.0

        # return dict(jpg=target, txt=prompt, hint=source)
        return {'im': target,  'sentence': prompt,'edge': source}