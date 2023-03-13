import logging
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

import numpy as np

from .mnist_utils import *
from .texture_utils import *

from .class_imbalance_utils import *
from .base_dataset import BaseDataset
from .image_utils import pil_loader

from .augmentation_utils import *

from global_config import BAR_CLASS_NAMES

class BARDataset(BaseDataset):
    def __init__(self,
            data_dir,
            split,
            transform=None,
            vis_transform=None,
            args=None,
            logits_file=None,
            minority_ratio=0.2,
            indices=None
        ):
        super(BARDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_classes = 6
        self.classes = []
        self.transform = transform
        self.images_dir = os.path.join(self.data_dir, self.split)
        self.logits_dir = logits_file
        self.minority_ratio = minority_ratio
        self.vis_transform = vis_transform
        if self.logits_dir is not None:
            self.create_groups()

        self.indices = indices
        self.prepare_dataset()

        if args:
            if args.epiwt:
                self.p_hats = np.ones((args.models_per_cycle, len(self), self.num_classes))
            else:
                self.p_hats = None

    def __len__(self):
        return len(self.ys)

    def prepare_dataset(self):
        filenames = list(sorted(os.listdir(self.images_dir)))
        self.unq_class_names = BAR_CLASS_NAMES
        self.ys = []
        self.file_names = []
        self.class_names = []
        self.class_ids = []
        self.item_ixs = []
        self.group_ixs = []

        for item_ix, fname in enumerate(filenames):
            class_name = fname.split('_')[0]
            self.ys.append(int(self.unq_class_names.index(class_name)))
            self.class_names.append(class_name)
            self.item_ixs.append(item_ix)
            if hasattr(self, 'minority_item_ixs'):
                is_minority = item_ix in self.minority_item_ixs
                self.group_ixs.append(is_minority)  # 0 = minority i.e., has high loss, 1 = majority
            else:
                self.group_ixs.append(item_ix % 2)  # Dummy odd/even group
            self.file_names.append(fname)

        self.ys = np.array(self.ys)

        # cut subset
        if self.indices:
            self.ys = self.ys[self.indices]
            self.file_names = np.array(self.file_names)[self.indices]
            self.group_ixs = np.array(self.group_ixs)[self.indices]
            self.item_ixs = np.array(self.item_ixs)[self.indices]
            self.class_names = np.array(self.class_names)[self.indices]


    def create_groups(self):
        results_holder = torch.load(self.logits_dir)
        ixs = torch.argsort(results_holder['losses'], descending=True)
        minority_ixs = ixs[:int(len(ixs) * self.minority_ratio)]
        self.minority_item_ixs = {}
        # sanity check -- do all classes fall under this minority?
        cls_to_minority = {}
        for ix in minority_ixs:
            y = int(results_holder['gt_labels'][ix])
            if y not in cls_to_minority:
                cls_to_minority[y] = 0
            cls_to_minority[y] += 1
            self.minority_item_ixs[int(results_holder['item_ixs'][ix])] = 1
        # self.minority_ixs = minority_ixs

    def getitems(self, select, vis=True):
        images = [self.__getitem__(s, vis)[0] for s in select]

        return torch.stack(images)

    def __getitem__(self, index, vis=False):
        fname = self.file_names[index]
        # img = pil_loader(os.path.join(self.images_dir, fname))
        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        # if self.transform is not None:

        if vis:
            img = self.vis_transform(img)
        else:
            img = self.transform(img)

        # item_data = {}
        # item_data['item_ix'] = index
        # item_data['x'] = img
        # item_data['y'] = self.ys[index]
        # item_data['class_name'] = self.class_names[index]
        # item_data['class_group_name'] = self.class_names[index]
        # item_data['group_ix'] = self.group_ixs[index]
        # item_data['file_name'] = self.file_names[index]
        return img, self.ys[index], index


def create_BAR_datasets(cfg, train_transform, test_transform, vis_transform, args):
    split_to_dataset = {'Test': {}}
    train_set = BARDataset(
        cfg.data_dir,
        'train',
        transform=train_transform,
        vis_transform=vis_transform,
        args=args,
        logits_file=cfg.logits_file
    )
    num_groups = len(train_set.class_names) * 2
    split_to_dataset['Train'] = train_set
    test_set = BARDataset(
        cfg.data_dir,
        'test',
        transform=test_transform,
        args=args,
        vis_transform=vis_transform
    )

    split_to_dataset['Test']['test_all'] = test_set
    # to keep validation loop; but cannot use for tuning.
    split_to_dataset['Test']['val'] = test_set

    for cls_name in BAR_CLASS_NAMES:
        select = np.where(np.array(test_set.class_names) == cls_name)
        split_to_dataset['Test'][f'{cls_name}'] = BARDataset(
            cfg.data_dir,
            'test',
            args=args,
            transform=test_transform,
            vis_transform=vis_transform,
            indices=select
        )

    # 0 = minority i.e., has high loss, 1 = majority
    is_minority = np.where(np.where(np.array(test_set.group_ixs), 0, 1))
    not_minority = np.where(np.array(test_set.group_ixs))

    split_to_dataset['Test']['test_minority'] = BARDataset(
        cfg.data_dir,
        'test',
        transform=test_transform,
        args=args,
        vis_transform=vis_transform,
        indices=is_minority
    )
    split_to_dataset['Test']['test_not_minority'] = BARDataset(
        cfg.data_dir,
        'test',
        transform=test_transform,
        vis_transform=vis_transform,
        indices=not_minority,
        args=args,
    )

    cfg.num_groups = num_groups

    return split_to_dataset

