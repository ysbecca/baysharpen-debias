import logging
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from .class_imbalance_utils import *
import pickle
from collections import namedtuple
from .augmentation_utils import build_transformation_list


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def preprocess_image(self, img):
        return img


if __name__ == "__main__":
    pass
