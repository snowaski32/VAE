import os
from torchvision.io import read_image
import cv2 as cv
import pytesseract as tes
import numpy as np
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset

load_path = "/home/ec2-user/.floorplans"
save_path = "images/"
data_name = "with_dilation"

patch_size = 512


class RPLanDataset(Dataset):
    def __init__(self, load_path, save_path, data_name, patch_size):
        self.load_path = load_path
        self.patch_size = patch_size
        self.data_name = data_name
        self.save_path = save_path

        self.toTensor = T.ToTensor()

        self.kernel = np.ones((3, 3), np.uint8)
        self.kernel2 = np.ones((3, 3), np.uint8)
        self.rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

        file_names = []
        if os.path.exists(save_path + "/list.txt"):
            with open(save_path + "/list.txt") as f:
                lines = f.read().split("\n")
                for line in lines:
                    if line != "":
                        file_names.append(line)
        else:
            with open(load_path + "/good_examples.txt") as f:
                lines = f.readlines()
                for line in lines:
                    if len(line.split()) == 2:
                        file_name, conf = line.split()
                        print(file_name)
                        if file_name.lower().endswith((".png", ".jpeg")):
                            img = read_image(load_path + "/plans/" + file_name)
                            if (
                                float(conf) >= 0.99
                                and img.shape[1] <= 600
                                and img.shape[2] <= 600
                            ):
                                file_names.append(file_name)

            with open(save_path + "/list.txt", "w") as f:
                f.writelines(name + "\n" for name in file_names)

        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.load_path + "/plans/" + self.file_names[idx]

        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = tes.image_to_boxes(image, output_type=tes.Output.DICT)
        if "char" in results:
            for j in range(0, len(results["char"])):
                if results["char"][j] != "~":
                    x = results["left"][j]
                    y = image.shape[0] - results["top"][j]

                    x2 = results["right"][j]
                    y2 = image.shape[0] - results["bottom"][j]

                    image = cv.rectangle(
                        image,
                        (x, y),
                        (x2, y2),
                        (int(image[0][0][0]), int(image[0][0][1]), int(image[0][0][2])),
                        -1,
                    )
        image = cv.resize(image, (patch_size, patch_size))
        image = cv.dilate(image, self.rect_kernel)
        image = cv.Canny(image, 100, 200)
        image = cv.dilate(image, self.rect_kernel)

        # image = cv.erode(image, self.kernel)

        image = self.toTensor(image)

        vutils.save_image(image, f"{self.save_path}/{self.data_name}/image{idx}.png")

        return path


dataset = RPLanDataset(
    load_path,
    save_path,
    data_name,
    patch_size,
)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=16,
    shuffle=False,
    pin_memory=False,
)

# next(iter(data_loader))
# dataset[32]
for path in data_loader:
    pass
