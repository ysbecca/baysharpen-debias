import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import math

import numpy as np
import pandas as pd

import torchvision.transforms.functional as F
from torch.utils.data import random_split
from scipy.ndimage.filters import gaussian_filter

from .coco_on_places import *
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


class DataModuleCocoPlaces(pl.LightningDataModule):

    def __init__(self, batch_size, args, dev_run=False):
        super().__init__()

        self.args                = args
        self.batch_size          = batch_size


        sizes = [LARGER_COCO_SIZE, LARGER_COCO_SIZE] if args.use_vw_flag_coco else [64, COCO_PLACES_IMSIZE]

        self.train_transform     = transforms.Compose([
            transforms.Resize(sizes[0]),
            transforms.RandomCrop(sizes[1], padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.test_transform      = transforms.Compose([
            transforms.Resize(sizes[0]),
            transforms.ToTensor(),
        ])


    def setup(self, stage=None):
        # converting from my args to their hydra config
        config = {
            'data_dir': COCO_PLACES_BASE_DIR,
            'num_workers': DEFAULT_NUM_WORKERS,
            'original_image_size': COCO_PLACES_ORIG_SIZE, 
            'image_size': COCO_PLACES_IMSIZE, 
            'name': 'coco_on_places', 
            'batch_size': self.batch_size, 
            'num_classes': NUM_CLASSES[COCO_PLACES_CODE], 
            'num_batches': math.ceil(BMNIST_TRAINSIZE / self.batch_size)
        }

        cfg = ConfObject(config)
        datasets = create_coco_on_places_datasets(cfg, self.train_transform, self.test_transform, self.args)

        self.train_set = datasets['Train']
        self.valid_set = datasets['Test']['validtest']
        
        # excluding anomalies for now
        set_aliases = ['idtest', 'validtest', 'oodtest', 'valoodtest', 'sgtest', 'sgval']

        # multiple test sets
        self.datasets = {key: datasets['Test'][key] for key in set_aliases}

        self.datasets['train'] = self.train_set
        self.datasets['val'] = self.valid_set

        self.keys = set_aliases + ['train', 'val']

        print("Coco-on-Places")
        print("___"*20)
        for key in self.keys:
            print(f"   > {key} set {len(self.datasets[key])}")

    def get_targets(self, key):
        return self.datasets[key].ys

    def get_select(self, key, select):
        """ returns selected images from set specified by key, with 
            relevant transform applied.
        """
        assert key in self.keys
        return self.datasets[key].getitems(select)

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

    def get_dataloader(self, key):
        assert key in self.keys
        return DataLoader(
            self.datasets[key],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
        )
