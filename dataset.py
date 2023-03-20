import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchvision 
from torchvision.datasets import CelebA
import zipfile
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
import cv2 as cv
from PIL import Image
from models import canny


# Add your custom dataset class here
class RPLanDataset(Dataset):
    def __init__(self, load_path, transforms, patch_size, split):
        self.load_path = load_path
        self.patch_size = patch_size

        self.transform1 = T.Resize((self.patch_size, self.patch_size))
        self.transform2 = T.ToTensor()

        file_names = []
        if os.path.exists(load_path + "/list.txt"):
            with open(load_path + "/list.txt") as f:
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

            print(len(file_names))
        
            with open(load_path + '/list.txt', 'w') as f:
                f.writelines(name + '\n' for name in file_names)

        # self.file_names = (
        #     file_names[: int(len(file_names) * 0.75)]
        #     if split == "train"
        #     else file_names[int(len(file_names) * 0.75) :]
        # )

        self.file_names = file_names[:1]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.load_path + "/plans/" + self.file_names[idx]

        # with open(path, "rb") as f:
        img = default_loader(path)
        # img = cv.Canny(img, 100, 200)
        img = self.transform2(img)
        img = self.transform1(img)
        return img, 0.0
        # if img.shape[0] == 2:
        #     img = torch.unsqueeze(img[0], dim=0)
        #     img = torch.concatenate([img]*3, axis=0)
        # elif img.shape[0] == 4:
        #     img = img[:3, : :]
        # elif img.shape[0] == 1:
        #     img = torch.concatenate([img]*3, axis=0)
        



class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self, data_path: str, split: str, transform: Callable, **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == ".jpg"])

        self.imgs = (
            imgs[: int(len(imgs) * 0.75)]
            if split == "train"
            else imgs[int(len(imgs) * 0.75) :]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        save_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        print(patch_size)

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.save_path = save_path

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  OxfordPets Dataset  =========================

        #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                               transforms.CenterCrop(self.patch_size),
        # #                                               transforms.Resize(self.patch_size),
        #                                               transforms.ToTensor(),
        #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                             transforms.CenterCrop(self.patch_size),
        # #                                             transforms.Resize(self.patch_size),
        #                                             transforms.ToTensor(),
        #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        #         self.train_dataset = OxfordPets(
        #             self.data_dir,
        #             split='train',
        #             transform=train_transforms,
        #         )

        #         self.val_dataset = OxfordPets(
        #             self.data_dir,
        #             split='val',
        #             transform=val_transforms,
        #         )

        #       =========================  CelebA Dataset  =========================

        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.patch_size),

            ]
        )

        self.train_dataset = RPLanDataset(
            self.data_dir,
            transforms,
            self.patch_size,
            split="train",
        )

        # Replace CelebA with your dataset
        self.val_dataset = RPLanDataset(
            self.data_dir,
            transforms,
            self.patch_size,
            split="test",
        )

        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )

        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
