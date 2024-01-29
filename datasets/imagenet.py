import torch
import torchvision
from torchvision.transforms import v2

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageNetDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = '/data/imagenet/imagenet', batch_size=64, num_workers=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transforms = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir, split='train', transform=self.train_transforms)
        self.val_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir, split='val', transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)