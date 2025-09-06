import json
import cv2
import os
from basicsr.utils import img2tensor
import numpy as np
import torch


class dataset_coco_mask_color():
    def __init__(self, path_json, root_path_im, image_size):#__init__(self, path_json, root_path_im, root_path_mask, image_size):
        super(dataset_coco_mask_color, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        # self.root_path_mask = root_path_mask
        for file in data:
            name = "%012d.png" % file['image_id']
            self.files.append({'name': name, 'sentence': file['caption']})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', '.jpg')))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        # mask = cv2.imread(os.path.join(self.root_path_mask, name))  # [:,:,0]
        # mask = cv2.resize(mask, (512, 512))
        # mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im': im,  'sentence': sentence} # {'im': im, 'mask': mask, 'sentence': sentence}

    def __len__(self):
        return len(self.files)
class dataset_coco_sketch():
    def __init__(self, path_json, root_path_im, root_path_sketch, image_size):
        super(dataset_coco_sketch, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_sketch = root_path_sketch
        for file in data:
            name = "%012d.png" % file['image_id']
            self.files.append({'name': name, 'sentence': file['caption']})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', '.jpg')))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        sketch = cv2.imread(os.path.join(self.root_path_sketch, name), cv2.IMREAD_GRAYSCALE) 
        sketch = cv2.resize(sketch, (512, 512))
        # 使用OpenCV的threshold函数进行二值化，确保只有纯黑和纯白
        _, sketch = cv2.threshold(sketch, 127, 255, cv2.THRESH_BINARY)

        # 转换为浮点数并归一化到0-1范围
        sketch = sketch.astype(np.float32) / 255.0

        # 添加通道维度
        sketch = np.expand_dims(sketch, axis=0)

        # 转换为PyTorch张量
        sketch = torch.from_numpy(sketch)
        

        sentence = file['sentence']
        return {'im': im, 'sketch': sketch, 'sentence': sentence}

    def __len__(self):
        return len(self.files)
class dataset_coco():
    def __init__(self, root_path_im):
        super(dataset_coco, self).__init__()
        self.files = []
        self.root_path_im = root_path_im
        # 遍历root_path_im目录中的文件名并追加到self.files列表中
        if os.path.exists(self.root_path_im):
            for filename in os.listdir(self.root_path_im):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    name = filename.replace('.png', '.jpg').replace('.jpeg', '.jpg')
                    self.files.append({'name': name})
    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', '.jpg')))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.
        return {'name': name, 'im': im}
    def __len__(self):
        return len(self.files)