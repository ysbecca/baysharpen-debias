"""

Biased synbols datasets.


"""


import torch


from torchvision.datasets import ImageFolder
import os
from skimage.filters import gaussian
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image

import numpy as np
from global_config import *

def get_sp_mask(spurious_mode, dice, sf_size):
	" return the mask for the regional SF "

	box_size = sf_size
	mask = np.zeros((IMAGENET_IMSIZE, IMAGENET_IMSIZE))
	if dice == 0:
		mask[:box_size, :box_size] = 1
	elif dice == 1:
		mask[:box_size, (IMAGENET_IMSIZE - box_size):IMAGENET_IMSIZE] = 1
	elif dice == 2:
		mask[(IMAGENET_IMSIZE - box_size):IMAGENET_IMSIZE, :box_size] = 1
	else:
		mask[(IMAGENET_IMSIZE - box_size):IMAGENET_IMSIZE, (IMAGENET_IMSIZE - box_size):IMAGENET_IMSIZE] = 1

	return mask

def add_sf(x, spurious_mode, dice=0, masks=None, sf_size=MASK_SIZE):
	" adds spurious feature depending on mode to list of images "

	if spurious_mode == SPURIOUS_MODE_REG:
		# insert black square into random corner of each image

		if dice == 0: # top left
			x[:, :sf_size, :sf_size, :] = (0, 0, 0)
		elif dice == 1: # top right
			x[:, :sf_size, (IMAGENET_IMSIZE - sf_size):IMAGENET_IMSIZE, :] = (0, 0, 0)
		elif dice == 2: # bottom left
			x[:, (IMAGENET_IMSIZE - sf_size):IMAGENET_IMSIZE, :sf_size, :] = (0, 0, 0)
		else: # bottom right
			x[:, (IMAGENET_IMSIZE - sf_size):IMAGENET_IMSIZE, (IMAGENET_IMSIZE - sf_size):IMAGENET_IMSIZE, :] = (0, 0, 0)

	elif spurious_mode == SPURIOUS_MODE_TEX:
		# colour digit solid black
		idxes = (masks == 1.)
		x[idxes] = (0, 0, 0)

	elif spurious_mode == SPURIOUS_MODE_QUAL:
		# reduce quality by gaussian blur
		x = np.array([gaussian(x_, sigma=(3, 3), truncate=3.5, multichannel=True)*255 for x_ in x])
		x = x.astype(np.uint8)

	elif spurious_mode == SPURIOUS_MODE_BW:
		# black and white
		x[:,:,:, 1] = x[:,:,:, 0]
		x[:,:,:, 2] = x[:,:,:, 0]

	elif spurious_mode == SPURIOUS_MODE_REG_ST:
		# regional stationary
		# print("REGIONAL STATIONARY!")
		x[:, :sf_size, :sf_size, :] = (0, 0, 0)

	else:
		print("[ERR] spurious mode not recognised,", spurious_mode)

	return x


class SpuriousSynbols(Dataset):
	def __init__(self, transform,
		bg_mode=GRAD_BG,
		spurious_ratio=DEFAULT_SPURIOUS_RATIO,
		spurious_mode=None,
		split="train",
		dev_run=False,
		seg_mode=False,
        sf_size=MASK_SIZE,
	):
		self.bg_mode 		= bg_mode
		self.transform 		= transform
		self.spurious_ratio = spurious_ratio
		self.spurious_mode 	= spurious_mode
		self.seg_mode  		= seg_mode
		self.sf_size 		= sf_size

		cap = 2000 if split == "train" else 1000
		if dev_run:
			cap = 250

		# load the images from npy files
		x = np.concatenate(
			(
				np.load(f"{SYNBOLS_BASE_DIR}{self.bg_mode}_t_{split}_x.npy")[:cap],
				np.load(f"{SYNBOLS_BASE_DIR}{self.bg_mode}_s_{split}_x.npy")[:cap]
			),
			axis=0,
		)
		y = np.concatenate(
			(
				np.load(f"{SYNBOLS_BASE_DIR}{self.bg_mode}_t_{split}_y.npy")[:cap],
				np.load(f"{SYNBOLS_BASE_DIR}{self.bg_mode}_s_{split}_y.npy")[:cap]
			),
			axis=0,
		)

		digit_masks = np.concatenate(
		(
			np.load(f"{SYNBOLS_BASE_DIR}{self.bg_mode}_t_{split}_mask.npy")[:cap],
			np.load(f"{SYNBOLS_BASE_DIR}{self.bg_mode}_s_{split}_mask.npy")[:cap]
		),
		axis=0,
		)
		self.digit_masks = digit_masks

		if self.spurious_ratio > 0.0:

			dice = np.random.randint(4, size=1)
			if self.spurious_ratio == 1.0:
				x = add_sf(x, self.spurious_mode, dice=dice)
				self.has_sp_feature = np.ones(len(y))
				# print("(SF) all dataset has spurious feature")

			else:
				has_sp_feature = np.zeros(len(y))
				class_len = int(len(x) / 2.)
				t_cutoff = int(spurious_ratio * class_len)
				s_cutoff = class_len - t_cutoff

				x[:t_cutoff] = add_sf(
					x[:t_cutoff],
					self.spurious_mode,
					dice=dice,
					masks=self.digit_masks[:t_cutoff],
					sf_size=self.sf_size,
				)
				x[class_len:(class_len + s_cutoff)] = add_sf(
					x[class_len:(class_len + s_cutoff)],
					self.spurious_mode,
					dice=dice,
					masks=self.digit_masks[class_len:(class_len + s_cutoff)],
					sf_size=self.sf_size,
				)

				has_sp_feature[:t_cutoff] = 1
				has_sp_feature[class_len:(class_len + s_cutoff)] = 1

				self.has_sp_feature = has_sp_feature
				self.spurious_mask 	= get_sp_mask(self.spurious_mode, dice, self.sf_size)
		else:
			# print("(CF) dataset with only core features")
			# assume no spurious features
			self.has_sp_feature = np.zeros(len(x))

		# TODO this is not working; for sharpening we do need a shuffle!
		# if split == "val":
		# 	# shuffle validation set
		# 	print(f"[INFO] Shuffling validation set...")
		# 	arr = np.arange(len(x))
		# 	np.random.shuffle(arr)
		# 	x = x[arr]
		# 	y = y[arr]
		#
		# 	self.has_sp_feature = self.has_sp_feature[arr]
		# 	if self.seg_mode:
		# 		self.digit_masks = self.digit_masks[arr]


		self.images 	   	= x
		self.targets 	 	= y
		self.has_sp_feature = self.has_sp_feature.astype(int)

		print(f"Set has {np.count_nonzero(np.where(self.has_sp_feature & self.targets, 1, 0))} T's w SF")
		print(f"Set has {np.count_nonzero(np.where(self.has_sp_feature & np.where(self.targets, 0, 1), 1, 0))} S's w SF")


	def __len__(self):
		return len(self.targets)


	def __getitem__(self, index):

		x = Image.fromarray(self.images[index])
		x = self.transform(x)

		if self.seg_mode:
		    # seg is a tensor assigning semantic segmentation labels to every
		    # pixel of an image seg[i, 0]
        # plot group epis...is 0th label for each pixel of ith image
		    # (2, y, x)

		    # 0 index : spurious feature
		    # 1 index : the true class mask
		    seg = torch.zeros((2, IMAGENET_IMSIZE, IMAGENET_IMSIZE))
		    if self.has_sp_feature[index]:
			    seg[0] = torch.tensor(self.spurious_mask)

		    seg[1] = torch.tensor(self.digit_masks[index])

		    return seg.int(), x
		else:
			return x, self.targets[index], index
