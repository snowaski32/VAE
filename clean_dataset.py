import os
from torchvision.io import read_image
import torch
import torchvision

dataset_name = 'with_dilation'

count = 0
for file in os.listdir(f"images/{dataset_name}/"):
    if 'png' in file:
        path = f'images/{dataset_name}/{file}'
        img = read_image(path, torchvision.io.ImageReadMode.GRAY)

        if torch.mean(img, dtype=float) < 5.0:
            os.remove(path)
            count += 1

print(count)