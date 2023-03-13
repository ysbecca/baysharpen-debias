import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import math

import numpy as np
import pandas as pd

import torchvision.transforms.functional as F
from torch.utils.data import random_split
from scipy.ndimage.filters import gaussian_filter

from .biased_mnist import *
from .data_utils import *

from global_config import *



def dict_collate_fn():
    return Compose([
        ListDictsToDictLists(),
        StackTensors()
    ])

class ConfObject:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


class DataModuleBiasedMNIST(pl.LightningDataModule):

    def __init__(self, batch_size, args, dev_run=False):
        super().__init__()

        self.args                = args

        self.batch_size          = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize(self.args.bmnist_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        )

        self.test_transform = transforms.Compose([
            transforms.Resize(self.args.bmnist_size),
            transforms.ToTensor()]
        )


    def setup(self, stage=None):


        # converting from my args to their hydra config
        config = {
            'data_dir': BMNIST_BASE_DIR,
            'num_workers': DEFAULT_NUM_WORKERS,
            'num_groups': None,
            'bias_split_name': 'full',
            'trainval_sub_dir': f'full_{self.args.spurious_ratio}', 
            'sampling_type': 'default', 
            'sampling_gamma': 0.5, 
            'sampling_attributes': None, 
            'normalize': False, 
            'original_image_size': BMINST_IMAGE_SIZE, 
            'image_size': BMINST_IMAGE_SIZE, 
            'augmentations': None, 
            'name': 'biased_mnist', 
            'batch_size': self.batch_size, 
            'num_classes': NUM_CLASSES[BMNIST_CODE], 
            'bias_variables': ['digit'] + BMNIST_BIAS_VARS,
            'p_bias': self.args.spurious_ratio,
            'target_name': 'digit',
            'train_pct': None,
            'dataset_size': BMNIST_TRAINSIZE,
            'num_batches': math.ceil(BMNIST_TRAINSIZE / self.args.batch_size)
        }

        cfg = ConfObject(config)
        datasets = create_biased_mnist_datasets(cfg, self.train_transform, self.test_transform, self.args)

        self.train_set = datasets['Train']
        self.valid_set = datasets['Test']['Val']
        self.test_set = datasets['Test']['Test']

        self.datasets = {
            'train': self.train_set,
            'val': self.valid_set,
            'test': self.test_set,      
        }

        bias_keys = []

        # indices = self.test_set.get_indices(bias_vars)
        # for bias_var in BMNIST_BIAS_VARS:
        #     self.datasets[bias_var + "_maj"] = get_test_bmnist(BMNIST_BASE_DIR, self.test_transform, indices[bias_var + "_maj"])
        #     self.datasets[bias_var + "_min"] = get_test_bmnist(BMNIST_BASE_DIR, self.test_transform, indices[bias_var + "_min"])
        #     bias_keys += [bias_var + "_min", bias_var + "_maj"]

        # self.keys = ['train', 'val', 'test'] + bias_keys
        self.keys = ['train', 'val', 'test']

        print("BIASED MNIST")
        print("___"*20)
        for key in self.keys:
            print(f"    {key} set {len(self.datasets[key])}")


    def get_targets(self, key):
        return self.datasets[key].get_targets()

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

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
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
