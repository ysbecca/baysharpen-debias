import torch 


from torchvision.datasets import ImageFolder
import os
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict

from PIL import Image

import numpy as np
from global_config import *


# return (image, class_id, mask, feature_id) for all images
class SalientImageNetComplete(Dataset):
    def __init__(self, transform, mask_transform, dev_run=False, mask_threshold=0.5, seg_mode=False):
        self.dev_run   = dev_run
        self.transform = transform
        self.mask_transform = mask_transform
        self.seg_mode = seg_mode
        self.mask_threshold = int(mask_threshold * 255)


        wordnet_dict = eval(open(os.path.join(SIMNET_DIR, 'wordnet_dict.py')).read())
        
        images = []
        masks = []
        class_ids = []
        feature_ids = []

        for class_index in range(len(wordnet_dict)):
            wordnet_id = wordnet_dict[class_index]
        
            # get directories for images and masks of that class id
            class_im_path = os.path.join(IMAGENET_DIR, 'train', wordnet_id)
            class_masks_path = os.path.join(SIMNET_DIR, wordnet_id)

            # list of top 65 image names for that class with one of top 5 features
            image_names_file = os.path.join(class_masks_path, 'image_names_map.csv')
            image_names_df = pd.read_csv(image_names_file)
            
            # fetch the list of features for the class
            feature_indices = np.array([int(col) for col in list(image_names_df.columns)])

            if self.dev_run:
                feature_indices = feature_indices[:1]

            # print(f"class index: {class_index}   feature ids: {feature_indices}")
            for feature_index in feature_indices:
                image_names_feature = image_names_df[str(feature_index)].to_numpy()
                # load exactly one image per unique (cid, fid) if dev_run
                if self.dev_run:
                    image_names_feature = image_names_feature[:1] 

                if self.seg_mode:
                    # add exactly once
                    images.append(f"{class_im_path}/{image_names_feature[0]}.JPEG")

                # for all top 65 images for given feature, save image name
                for i, image_name in enumerate(image_names_feature):
                    mask_list = []
                    fid_list = []
                    if not self.seg_mode:
                        # with duplicates
                        images.append(f"{class_im_path}/{image_name}.JPEG")
                        # save the mask name for that image
                        masks.append(f"{SIMNET_DIR}{wordnet_id}/feature_{str(feature_index)}/{image_name}.JPEG")
                        class_ids.append(class_index)
                        feature_ids.append(feature_index)
                    else:
                        # save a list of masks and fids.
                        mask_list.append(f"{SIMNET_DIR}{wordnet_id}/feature_{str(feature_index)}/{image_name}.JPEG")
                        fid_list.append(feature_index)

                if self.seg_mode:
                    # masks and fids for that particular image
                    masks.append(mask_list)
                    feature_ids.append(fid_list)

        
        self.images = images
        self.masks = masks
        self.class_ids = class_ids
        self.feature_ids = feature_ids

        self.total_fids = 5000

    def __len__(self):
        # full duplicated length
        return len(self.images)


    def harden_mask(self, mask):
        m = np.array(mask)
        return Image.fromarray(np.where(m > self.mask_threshold, 255, 0).astype(np.uint8))

    def __getitem__(self, index):

        # get the mask to be 
        if not self.seg_mode:
            image_path = self.images[index]

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            
            mask = Image.open(self.masks[index])

            # hard binary mask
            mask = self.harden_mask(mask)


            mask_tensor = self.mask_transform(mask)

            return (image_tensor, mask_tensor, self.class_ids[index], self.feature_ids[index])
        else:
            seg = torch.zeros(self.total_fids, IMAGENET_IMSIZE, IMAGENET_IMSIZE)

            for fid, mask in zip(self.feature_ids[index], self.masks[index]):
                mask = Image.open(mask)
                mask = self.harden_mask(mask)

                # turn soft mask into a hard mask?
                # resize and convert to tensor
                mask = self.mask_transform(mask)

                # set the seg value for that fid to the mask
                seg[fid] = mask

            image_tensor = Image.open(self.images[index]).convert("RGB")
            image_tensor = self.transform(image_tensor)
            
            # seg is a tensor assigning semantic segmentation labels to every
            # pixel of an image seg[i, 0] is 0th label for each pixel of ith image
            # (N, fid, y, x) ==> return shape (fid, y, x)
            return seg.int(), image_tensor






def dilate_erode(masks, dilate=True, iterations=15, kernel=5):
    ''' Dilate or erode tensor of soft segmentation masks '''

    assert kernel % 2 == 1
    half_k = kernel // 2
    batch_size, _, side_len, _ = masks.shape

    out = masks[:,0,:,:].clone()
    padded = torch.zeros(batch_size, side_len+2*half_k, side_len+2*half_k, device=masks.device)
    if not dilate:
        padded = 1 + padded
    for itr in range(iterations):
        all_padded = []
        centered = padded.clone()
        centered[:, half_k:half_k+side_len, half_k:half_k+side_len]; all_padded.append(centered)
        for j in range(1, half_k+1):
            left, right, up, down = [padded.clone() for _ in range(4)]
            left[:, half_k-j:half_k-j+side_len, half_k:half_k+side_len] = out; all_padded.append(left)
            right[:, half_k+j:half_k+j+side_len, half_k:half_k+side_len] = out; all_padded.append(right)
            up[:, half_k:half_k+side_len, half_k+j:half_k+j+side_len] = out; all_padded.append(up)
            down[:, half_k:half_k+side_len, half_k-j:half_k-j+side_len] = out; all_padded.append(down)
    
        all_padded = torch.stack(all_padded)
        out = torch.max(all_padded, dim=0)[0] if dilate else torch.min(all_padded, dim=0)[0]
        out = out[:, half_k:half_k+side_len, half_k:half_k+side_len]

    out = torch.stack([out, out, out], dim=1)
    out = out / torch.max(out)
    return out




# Look at training images and masks by CLASS ID and FEATURE ID(s)
class SalientImageNet(Dataset):
    def __init__(self, masks_path, class_index, feature_indices, transform,
                 resize_size=256, crop_size=224):
        self.transform = transform

        wordnet_dict = eval(open(os.path.join(masks_path, 'wordnet_dict.py')).read())
        wordnet_id = wordnet_dict[class_index]
        
        self.images_path = os.path.join(IMAGENET_DIR, 'train', wordnet_id)
        self.masks_path = os.path.join(SIMNET_DIR, wordnet_id)
        
        image_names_file = os.path.join(self.masks_path, 'image_names_map.csv')
        print(image_names_file)
        image_names_df = pd.read_csv(image_names_file)
        
        print(image_names_df.head())

        image_names = []
        feature_indices_dict = defaultdict(list)
        for feature_index in feature_indices:
            image_names_feature = image_names_df[str(feature_index)].to_numpy()
            
            for i, image_name in enumerate(image_names_feature):
                image_names.append(image_name)                
                feature_indices_dict[image_name].append(feature_index)        
        
        self.image_names = np.unique(np.array(image_names))                
        self.feature_indices_dict = feature_indices_dict

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        curr_image_path = os.path.join(self.images_path, image_name + '.JPEG')

        image = Image.open(curr_image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        feature_indices = self.feature_indices_dict[image_name]
        
        all_mask = np.zeros(image_tensor.shape[1:])
        for feature_index in feature_indices:            
            curr_mask_path = os.path.join(self.masks_path, 'feature_' + str(feature_index), image_name + '.JPEG')
            
            mask = np.asarray(Image.open(curr_mask_path))
            mask = (mask/255.)
            
            all_mask = np.maximum(all_mask, mask)

        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)
        return image_tensor, mask_tensor