import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



import numpy as np
import pandas as pd

import torchvision.transforms.functional as F
from torch.utils.data import random_split
from scipy.ndimage.filters import gaussian_filter

from .synbols import *



from global_config import *



class DataModuleSpuriousSynbols(pl.LightningDataModule):

    def __init__(self, batch_size, seg_mode, args, mode="train"):
        super().__init__()

        self.batch_size          = args.batch_size
        self.bg_mode             = args.bg_mode
        self.spurious_ratio      = args.spurious_ratio
        self.spurious_mode       = args.spurious_mode
        self.dev_run             = args.dev_run
        self.sf_size             = args.sf_size
        self.seg_mode            = seg_mode
        self.mode                = mode

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def setup(self, stage=None):
        print("---- TRAIN")
        self.train_set = SpuriousSynbols(
            transform=self.transform,
            bg_mode=self.bg_mode,
            spurious_ratio=self.spurious_ratio,
            spurious_mode=self.spurious_mode,
            split="train",
            dev_run=self.dev_run,
            seg_mode=self.seg_mode,
            sf_size=self.sf_size
        )
        print("---- VALID: TD ")
        self.valid_set = SpuriousSynbols(
            transform=self.transform,
            bg_mode=self.bg_mode,
            spurious_ratio=self.spurious_ratio,
            spurious_mode=self.spurious_mode,
            split="val",
            dev_run=self.dev_run,
            seg_mode=self.seg_mode,
            sf_size=self.sf_size
        )
        # TOP is NOT equal to the BOTTOM even though the code is exactly the same...
        print("---- VALID: CF ")
        self.valid_set_cf = SpuriousSynbols(
            transform=self.transform,
            bg_mode=self.bg_mode,
            spurious_ratio=0.0,
            spurious_mode=self.spurious_mode,
            split="val",
            dev_run=self.dev_run,
            seg_mode=self.seg_mode,
            sf_size=self.sf_size
        )
        print("---- VALID: SF ")
        self.valid_set_sf = SpuriousSynbols(
            transform=self.transform,
            bg_mode=self.bg_mode,
            spurious_ratio=1.0,
            spurious_mode=self.spurious_mode,
            split="val",
            dev_run=self.dev_run,
            seg_mode=self.seg_mode,
            sf_size=self.sf_size
        )

        print("synbols train size", len(self.train_set))
        print("synbols valid size", len(self.valid_set), len(self.valid_set_cf), len(self.valid_set_sf))


    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=GPU_COUNT,
        )

    def val_dataloader_train(self, workers=DEFAULT_NUM_WORKERS):
        # set matching training distribution
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=GPU_COUNT,
        )

    # the default called by lightning; force return for now
    def val_dataloader(self):
        return self.val_dataloader_train()


    def val_cf_dataloader(self, workers=DEFAULT_NUM_WORKERS):
        return DataLoader(
            self.valid_set_cf,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=GPU_COUNT,
        )

    def val_sf_dataloader(self, workers=DEFAULT_NUM_WORKERS):
        return DataLoader(
            self.valid_set_sf,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=GPU_COUNT,
        )
