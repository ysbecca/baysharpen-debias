import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# for older PyTorch versions....
from .imagenet import ImageNet
# newer versions
# from torchvision.datasets import ImageNet

import numpy as np
import pandas as pd

import torchvision.transforms.functional as F
from torch.utils.data import random_split
from scipy.ndimage.filters import gaussian_filter

from .simnet import *



from global_config import *



class DataModuleSalientImageNet(pl.LightningDataModule):

    def __init__(self, batch_size, seg_mode=False, mask_threshold=0.5, dev_run=False):
        super().__init__()
          
        self.batch_size          = batch_size
        # control it here since iterate over full loader later
        self.dev_run             = dev_run
        self.seg_mode            = seg_mode
        self.mask_threshold      = mask_threshold

        self.eval_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGENET_IMSIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGENET_IMSIZE),
            transforms.ToTensor(),
        ])


    def setup(self, stage=None):

        self.simnet = SalientImageNetComplete(
            transform=self.eval_transform,
            mask_transform=self.mask_transform,
            dev_run=self.dev_run,
            seg_mode=self.seg_mode,
            mask_threshold=self.mask_threshold,
        )


        print("simnet size", len(self.simnet))


    def simnet_dataloader(self):
        return DataLoader(
            self.simnet,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=GPU_COUNT,
        )


class DataModuleImageNet(pl.LightningDataModule):

    def __init__(
            self,
            batch_size,
            small_imagenet=False,
            subset100=False,
        ):
        super().__init__()
          
        self.batch_size          = batch_size
        self.small_imagenet      = small_imagenet
        self.subset100           = subset100

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGENET_IMSIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGENET_IMSIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


    def setup(self, stage=None):

        self.train_set = ImageNet(
            root=IMAGENET_DIR,
            split="train",
            transform=self.train_transform,
            small_imagenet=self.small_imagenet,
            subset100=self.subset100,
        )

        self.valid_set = ImageNet(
            root=IMAGENET_DIR,
            split="val",
            transform=self.eval_transform,
            small_imagenet=self.small_imagenet,
            subset100=self.subset100,
        )

        print("train set size", len(self.train_set))
        print("valid set size", len(self.valid_set))

  
    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=GPU_COUNT
        )

  
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=GPU_COUNT,
        )


