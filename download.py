import os
from dagshub.streaming import DagsHubFilesystem
from PIL import Image

fs = DagsHubFilesystem('.', repo_url='https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus', branch="main")
fs.install_hooks()

files = fs.listdir('data/')
with fs.open('data/labels.tsv') as tsv:
    for row in tsv.readlines()[:100]:
        row = row.strip()
        try:
            img_file, caption, score, url = row.split('\t')
            img_path = os.path.join('data', img_file)
            img = Image.open(img_path)
            print(f'{img_file} 的尺寸为 {img.size}，美学评分为 {score}')
        except Exception as e:
            print(f'处理 {row} 时出错: {e}')
