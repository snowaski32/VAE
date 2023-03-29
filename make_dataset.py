import os
from torchvision.io import read_image
import cv2 as cv
import pytesseract as tes
import numpy as np
import torchvision.transforms as T
import torchvision.utils as vutils

load_path = "/home/ec2-user/.floorplans"
save_path = "images/"

patch_size = 512

kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

transform1 = T.Resize((patch_size, patch_size))
transform2 = T.ToTensor()

file_names = []

if os.path.exists(save_path + "/list.txt"):
    with open(save_path + "/list.txt") as f:
        lines = f.read().split('\n')
        for line in lines:
            if line != '':
                file_names.append(line)
else:
    with open(load_path + "/good_examples.txt") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split()) == 2:
                file_name, conf = line.split()
                print(file_name)
                if (
                    file_name.lower().endswith((".png", ".jpeg"))
                ):
                    img = read_image(load_path + "/plans/" + file_name)
                    if (
                        float(conf) >= 0.99
                        and img.shape[1] <= 600
                        and img.shape[2] <= 600
                    ):
                        file_names.append(file_name)

    with open(save_path + '/list.txt', 'w') as f:
        f.writelines(name + '\n' for name in file_names)

names = []
for i, f in enumerate(file_names):
    print(i)
    path = load_path + "/plans/" + f
   

    image = cv.imread(path)

    results = tes.image_to_boxes(image, output_type=tes.Output.DICT)

    if 'char' in results:
        for j in range(0, len(results['char'])):
            if results['char'][j] != '~':
                x = results['left'][j]
                y = image.shape[0] - results['top'][j]

                x2 = results['right'][j]
                y2 = image.shape[0] - results['bottom'][j]

                image = cv.rectangle(image, (x, y), (x2, y2), (int(image[0][0][0]), int(image[0][0][1]), int(image[0][0][2])), -1)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (patch_size, patch_size))
    # image = cv.dilate(image, kernel2)
    image = cv.Canny(image, 100, 200)
    # image = cv.dilate(image, kernel)
    image = transform2(image)

    vutils.save_image(image, f"{save_path}image{i}.png")
    names.append(f"image{i}.png")

with open(save_path + '/images.txt', 'w') as f:
    f.writelines(name + '\n' for name in names)
