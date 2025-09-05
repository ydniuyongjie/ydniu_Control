from ldm.modules.extra_condition.model_edge import pidinet
import torch
from ldm.data.dataset_coco import dataset_coco
import os
import cv2
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from tqdm import tqdm
# load model
net_G = pidinet()
ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
net_G.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
net_G.cuda()
dataset = dataset_coco(root_path_im='coco/train2017')
sketch_path="coco_stuff/sketch/train2017_sketch"
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

if not os.path.exists(sketch_path):
    os.makedirs(sketch_path)

for i, data in enumerate(tqdm(dataloader, desc="Processing images", total=len(dataloader))):
    edge = net_G(data['im'].cuda(non_blocking=True))[-1]
    edge = edge > 0.5
    edge = edge.float()
    im_edge = tensor2img(edge)
    name = data['name'][0]
    cv2.imwrite(os.path.join(sketch_path, name), im_edge)
    tqdm.write(f"Processed: {name}")  # 更新进度条信息
    
print("sketch completed.")   