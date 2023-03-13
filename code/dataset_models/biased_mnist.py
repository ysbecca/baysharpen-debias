import logging
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

from torchvision import datasets, transforms

from .mnist_utils import *
from .texture_utils import *

from .class_imbalance_utils import *
from .base_dataset import BaseDataset
from .image_utils import pil_loader

from .augmentation_utils import *
import torchvision.transforms.functional as tF



class BiasedMNISTDataset(BaseDataset):
    def __init__(self, data_dir=None,
                 bias_split_name='full',
                 trainval_sub_dir='full_0.95',
                 split='Test',
                 target_name='digit',
                 bias_variables=['digit', 'digit_color_ix', 'digit_scale_ix', 'digit_position_ix', 'texture_ix',
                                 'texture_color_ix', 'letter_ix', 'letter_color_ix'],
                 transform=transforms.Compose([
                     transforms.Resize(160),
                     # transforms.RandomHorizontalFlip(),
                     transforms.ToTensor()]
                 ),
                 joint_transform=None,
                 indices=None,
                 args=None,
            ):
        """

        :param data_dir: Directory where the images and factor info are located
        :param bias_split_name: Name of specific subset within biased mnist e.g., digit_vs_digit_color
        :param trainval_sub_dir: Name of the sub-split. Each bias_split (e.g., long_tailed_mnist) may contain multiple sub-splits e.g., long_tailed_mnist_0.01
        :param split: train or val or test
        :param target_name: What do want to predict? e.g., digit, texture_ix, color_ix
        :param bias_variables: Samples will be grouped using these variables.
        :param joint_transform: Transforms that apply to both image and mask
        :param transform: Transforms that apply only to mask
        """
        super(BiasedMNISTDataset, self).__init__()
        self.data_dir = data_dir
        self.bias_split_name = bias_split_name
        self.trainval_sub_dir = trainval_sub_dir
        self.split = split
        self.target_name = target_name
        self.bias_variables = bias_variables
        self.num_classes = 10

        # Load the data files
        self.indices = indices
        self.load_factor_config()
        self.prepare_dataset()

        self.num_groups = self.main_group_utils.max_group_ix
        self.joint_transform = joint_transform
        self.transform = transform

        if args:
            if args.epiwt:
                self.p_hats = np.ones((args.models_per_cycle, len(self), self.num_classes))
            else:
                self.p_hats = None

    def load_factor_config(self):
        if 'train' in self.split or 'val' in self.split:
            split_name = 'trainval'
            sub_dir = self.trainval_sub_dir
        else:
            split_name = 'test'
            sub_dir = f'{self.bias_split_name}'

        self.images_dir = os.path.join(self.data_dir, sub_dir, split_name)
        # print(self.images_dir, "-"*20)
        self.factors_data = json.load(open(os.path.join(self.data_dir, sub_dir, f'{split_name}.json')))
        if self.indices:
            self.factors_data = [self.factors_data[idx] for idx in self.indices]
        # if split_name == 'test':
        #     count_groups(self.factors_data)

        self.main_group_utils = GroupUtils(self.target_name, self.bias_variables)

    def __len__(self):
        return len(self.factors_data)

    def get_indices(self, bias_vars):

        indices = {}

        for bias_var in bias_vars:
            min_idxes, maj_idxes = [], []
            
            # digit_color_ix_group_name
            grp_name = bias_var + "_group_name"
            for idx in range(len(self)):
                x = getattr(self, grp_name)[idx]

                if x.endswith('_majority'):
                    maj_idxes.append(idx)
                else:
                    min_idxes.append(idx)

            indices[bias_var + "_maj"] = maj_idxes
            indices[bias_var + "_min"] = min_idxes

        return indices

    def prepare_dataset(self):
        """
        Assigns each data item to a group
        :return:
        """
        # self.data_items = {}
        self.data_item_keys = []
        self.ix_to_item_ix = {}
        self.item_ix_to_ix = {}
        self.class_ix_to_name = {}
        for index, curr_factor_to_val in enumerate(self.factors_data):
            curr_factor_to_val = self.factors_data[index]  # Contains exact values for each factor
            group_ix, group_name, maj_min_group_ix, maj_min_group_name = \
                self.main_group_utils.to_group_ix_and_name(curr_factor_to_val)

            # Gather the class id of the target attribute and add group based on bias variable
            y = curr_factor_to_val[self.target_name]
            self.class_ix_to_name[y] = str(y)
            item_data = {
                'y': y,
                'class_name': str(y),
                'item_ix': curr_factor_to_val['index'],
                'group_ix': group_ix,
                'index': index,
                'group_name': group_name,
                # 'maj_min_group_ix': maj_min_group_ix,
                # 'maj_min_group_name': maj_min_group_name
            }
            self.item_ix_to_ix[curr_factor_to_val['index']] = index
            self.ix_to_item_ix[index] = curr_factor_to_val['index']
            for k in curr_factor_to_val:
                if k.endswith('_ix'):
                    grp_key = f'{k}_group_name'
                    if y == curr_factor_to_val[k]:
                        item_data[grp_key] = k + '_majority'
                    else:
                        item_data[grp_key] = k + '_minority'
                    # item_data['group_name'] = item_data[grp_key]

            # There is a memory leak issue with copy-on-access with data types such as dict/list
            # Instead of using a dict, let us create separate numpy arrays per key
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

            for k in item_data.keys():
                if not hasattr(self, k):
                    setattr(self, k, [])
                    self.data_item_keys.append(k)
                getattr(self, k).append(item_data[k])
        for k in self.data_item_keys:
            setattr(self, k, np.asarray(getattr(self, k)))

    def get_targets(self, indices=None):
        if indices:
            return np.array([getattr(self, 'y')[idx] for idx in indices])
        else:
            return np.array([getattr(self, 'y')[idx] for idx in range(len(self))])

    def __getitem__(self, index):
        item_data = {}
        for k in self.data_item_keys:
            item_data[k] = getattr(self, k)[index]

        img = pil_loader(os.path.join(self.images_dir, f'{index}.jpg'))

        if self.transform is not None:
            img = self.transform(img)

        item_data['x'] = img
        item_data['y'] = torch.LongTensor([item_data['y']]).squeeze()

        return item_data['x'], item_data['y'], index
        
    def getitems(self, select):
        images = [self.__getitem__(s)[0] for s in select]

        return torch.stack(images)

def get_test_bmnist(base_dir, transform, indices):
    # test set subsets
    # sub_dir = f'full_0.95'
    sub_dir = f'full_0.5'
    bias_vars = [
                'digit',
                'digit_color_ix',
                'digit_scale_ix', 
                'digit_position_ix',
                'texture_ix',
                'texture_color_ix',
                'letter_ix',
                'letter_color_ix'
            ]
    dataset = BiasedMNISTDataset(base_dir, 'full',
                 sub_dir, 'test',
                 target_name='digit',
                 bias_variables=bias_vars,
                 args=None,
                 transform=transform,
                 indices=indices,
            )
    return dataset



def create_biased_mnist_dataset_for_split(dataset_cfg, split, transform, args):
    # first see if train_ixs are defined within the sub_dir, else use the default one
    if os.path.exists(os.path.join(dataset_cfg.data_dir, dataset_cfg.trainval_sub_dir, 'train_ixs.json')):
        train_ixs = json.load(open(os.path.join(dataset_cfg.data_dir, dataset_cfg.trainval_sub_dir, 'train_ixs.json')))
        val_ixs = json.load(open(os.path.join(dataset_cfg.data_dir, dataset_cfg.trainval_sub_dir, 'val_ixs.json')))
    else:
        train_ixs = json.load(open(os.path.join(dataset_cfg.data_dir, 'train_ixs.json')))
        val_ixs = json.load(open(os.path.join(dataset_cfg.data_dir, 'val_ixs.json')))
    if dataset_cfg.train_pct is not None:
        train_ixs = json.load(
            open(os.path.join(dataset_cfg.data_dir, f'subset_{dataset_cfg.train_pct}_train_ixs.json')))

    if 'train' in split.lower():
        # train_list, joint_list = build_transformation_list(dataset_cfg.augmentations, image_size=dataset_cfg.image_size)
        # train_transform, joint_transform = transforms.Compose(train_list), transforms.Compose(joint_list)
        train_set = BiasedMNISTDataset(dataset_cfg.data_dir, dataset_cfg.bias_split_name,
                                       dataset_cfg.trainval_sub_dir, 'train', target_name=dataset_cfg.target_name,
                                       bias_variables=dataset_cfg.bias_variables,
                                       transform=transform,
                                       args=args,
                                       indices=train_ixs)
        num_groups = train_set.num_groups
        dataset_cfg.num_groups = num_groups

        return train_set
    else:
        # test_list, joint_list = build_transformation_list(image_size=dataset_cfg.image_size)
        # test_transform, joint_transform = transforms.Compose(test_list), transforms.Compose(joint_list)
        sub_dir = dataset_cfg.trainval_sub_dir if split != 'val' else f'{dataset_cfg.bias_split_name}_0.5'

        if split.lower() == 'val':
            dataset = BiasedMNISTDataset(dataset_cfg.data_dir, dataset_cfg.bias_split_name,
                             sub_dir, split,
                             target_name=dataset_cfg.target_name,
                             bias_variables=dataset_cfg.bias_variables,
                             transform=transform,
                             args=args,
                             indices=val_ixs)
        else:
            dataset = BiasedMNISTDataset(dataset_cfg.data_dir, dataset_cfg.bias_split_name,
                             sub_dir, split,
                             target_name=dataset_cfg.target_name,
                             bias_variables=dataset_cfg.bias_variables,
                             args=args,
                             transform=transform)
        return dataset


def create_biased_mnist_datasets(dataset_cfg, train_transform, test_transform, args):

    split_to_dataset = {'Test': {}}
    split_to_dataset['Train'] = create_biased_mnist_dataset_for_split(dataset_cfg, 'train', train_transform, args)
    split_to_dataset['Test']['Val'] = create_biased_mnist_dataset_for_split(dataset_cfg, 'val', test_transform, args)
    split_to_dataset['Test']['Test'] = create_biased_mnist_dataset_for_split(dataset_cfg, 'test', test_transform, args)

    return split_to_dataset


def create_biased_mnist_dataloader_for_split(dataset_cfg, split):
    dataset = create_biased_mnist_dataset_for_split(dataset_cfg, split)
    if 'train' in split.lower():
        return DataLoader(
            dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=True,
            num_workers=dataset_cfg.num_workers,
            collate_fn=dict_collate_fn()
        )
    else:
        return DataLoader(
            dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=False,
            num_workers=dataset_cfg.num_workers,
            collate_fn=dict_collate_fn()
        )


def create_biased_mnist_dataloaders(config):
    ds_cfg = config.dataset
    logging.getLogger().info(f"Setting the num_groups to {ds_cfg.num_groups}")
    out = {'test': {}, 'val': {}}
    out['train'] = create_biased_mnist_dataloader_for_split(ds_cfg, 'train')
    out['val']['val'] = create_biased_mnist_dataloader_for_split(ds_cfg, 'val')
    out['test']['test'] = create_biased_mnist_dataloader_for_split(ds_cfg, 'test')
    return out


class GroupUtils():
    def __init__(self, target_name, bias_variables, num_classes=10, use_majority_minority_grouping=False):
        """
        Groups the data based on the specified bias variables i.e., each unique combination of bias variables is a group
        e.g., group#1 = (digit color=red, texture=+, texture color=green etc.)

        If use_majority_minority_grouping is used

        :param target_name: Variable to predict
        :param bias_variables: List of variables that act as biases
        :param num_classes:
        :param target_name:
        :param use_majority_minority_grouping:
        """
        self.num_classes = num_classes
        self.target_name = target_name
        self.bias_variables = bias_variables
        self.use_majority_minority_grouping = use_majority_minority_grouping
        self.group_name_to_ix = {}
        self.maj_min_group_name_to_ix = {}
        self.max_group_ix = 0
        self.max_maj_min_group_ix = 0

    def to_group_ix_and_name(self, curr_factor_to_val):
        group_name_parts = []

        # Assume that if the factor ix is same as the index of the factor val, then it is a majority group,
        # else it is a minority group
        maj_min_group_name_parts = []
        class_ix = curr_factor_to_val[self.target_name]

        # Go through all of the bias variables, to come up with the group name
        for ix, bias_name in enumerate(self.bias_variables):
            bias_val_ix = curr_factor_to_val[bias_name]
            maj_min = 'minority'
            if bias_val_ix == class_ix:
                maj_min = 'majority'  # There is no majority/minority for lbl, so we just use 'majority' for label
            group_name_parts.append(f'{bias_name}_{bias_val_ix}')
            maj_min_group_name_parts.append(f'{bias_name}_{maj_min}')

        group_name = '+'.join(group_name_parts)
        maj_min_group_name = '+'.join(maj_min_group_name_parts)
        if self.use_majority_minority_grouping:
            group_name = maj_min_group_name

        if group_name not in self.group_name_to_ix:
            self.group_name_to_ix[group_name] = self.max_group_ix
            self.max_group_ix += 1
        if maj_min_group_name not in self.maj_min_group_name_to_ix:
            self.maj_min_group_name_to_ix[maj_min_group_name] = self.max_maj_min_group_ix
            self.max_maj_min_group_ix += 1
        group_ix = self.group_name_to_ix[group_name]
        maj_min_group_ix = self.maj_min_group_name_to_ix[maj_min_group_name]
        return group_ix, group_name, maj_min_group_ix, maj_min_group_name
