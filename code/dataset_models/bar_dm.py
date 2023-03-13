import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import math

import numpy as np
import pandas as pd

import torchvision.transforms.functional as F
from torch.utils.data import random_split
from scipy.ndimage.filters import gaussian_filter

from .bar import *
from .data_utils import *

from global_config import *


class ConfObject:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


class DataModuleBAR(pl.LightningDataModule):

    def __init__(self, batch_size, args, dev_run=False):
        super().__init__()

        self.args                = args

        self.batch_size          = batch_size

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                (BAR_IMAGE_SIZE, BAR_IMAGE_SIZE),
                scale=(0.7, 1.0),
                ratio=(1.0, 4./3),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )

        self.test_transform = transforms.Compose([
            transforms.Resize((BAR_IMAGE_SIZE, BAR_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )

        self.vis_transform = transforms.Compose([
            transforms.Resize((BAR_IMAGE_SIZE, BAR_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])


    def setup(self, stage=None):
        config = {
            'data_dir': BAR_BASE_DIR,
            'num_workers': DEFAULT_NUM_WORKERS,
            'augmentations': None,
            'name': 'biased_mnist',
            'logits_file': None,
            'batch_size': self.batch_size,
            'num_classes': NUM_CLASSES[BAR_CODE],
        }

        cfg = ConfObject(config)
        datasets = create_BAR_datasets(
            cfg,
            self.train_transform,
            self.test_transform,
            self.vis_transform,
            self.args,
        )
        self.keys = ['test_all', 'test_minority', 'test_not_minority']
        self.keys += BAR_CLASS_NAMES

        self.train_set = datasets['Train']
        self.valid_set = datasets['Test']['val']

        self.datasets = {
            'train': self.train_set,
            'val': self.valid_set
        }

        for key in self.keys:
            self.datasets[key] = datasets['Test'][key]
        self.keys += ['train', 'val']

        print("BAR")
        print("___"*20)
        for key in self.keys:
            print(f"    {key} set {len(self.datasets[key])}")


    def get_select(self, key, select, vis=False):
        """ returns selected images from set specified by key, with 
            relevant transform applied.
        """
        assert key in self.keys
        return self.datasets[key].getitems(select, vis=vis)


    def get_targets(self, key):
        return self.datasets[key].ys

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=DEFAULT_NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
        )

    def get_dataloader(self, key=None):
        assert key in self.keys
        return DataLoader(
            self.datasets[key],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
        )
