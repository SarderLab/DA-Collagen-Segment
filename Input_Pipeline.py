# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:39:49 2024

@author: fafsari

Data input pipeline for Deep-DA-DUET

from: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
"""

# import os
# import pandas as pd
import numpy as np
from math import floor, ceil

# from tqdm import tqdm
# import matplotlib.pyplot as plt
from skimage.io import imread#, imsave
# from glob import glob

# from random import sample

import torch
from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor
import albumentations

from skimage.transform import resize

from Augmentation_Functions import *
from Utils import resize_special

# Input class with required len and getitem functions
# in this case, the inputs and targets are lists of paths matching paths for image and segmentation ground-truth
# *** Have to make sure the input and target paths are aligned ***
# Need to change shape of input data to be (batch size, channels, height, width)
# with linear scaling depending on network expectations,
# target to be (batch size, height, width) with dense integer encoding for each class

class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 pre_transform=None,
                 target_type=None,
                 batch_size=None,
                 parameters={}):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.pre_transform = pre_transform
        self.inputs_dtype = torch.float32
        self.parameters = parameters

        if target_type == 'binary':
            self.targets_dtype = torch.long
        elif target_type == 'nonbinary':
            self.targets_dtype = torch.float32
        elif target_type is None:
            self.targets_dtype = torch.float32

        self.batch_size = batch_size
        self.patch_batch = False
        self.sample_weight = False

        self.testing_metrics = False if len(self.targets) == 0 else True

        image_size = [int(i) for i in self.parameters['preprocessing']['image_size'].split(',')]
        self.image_size = image_size
        self.original_size = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        img_name = self.inputs[index]
        tar_name = self.targets[index] if len(self.targets) > 0 else None

        # Read image(s)
        try:
            # For multi-channel image inputs
            if isinstance(img_name, list):
                img1 = imread(str(img_name[0]))
                img2 = imread(str(img_name[1]))
                img = np.concatenate((img1, img2), axis=-1)
                img_name_str = img_name[0]
            else:
                img = imread(str(img_name))
                img_name_str = img_name

            if tar_name is not None:
                tar = imread(str(tar_name))
            else:
                tar = np.zeros((img.shape[0], img.shape[1], 1))  
            self.original_size = (img.shape[0], img.shape[1])      
        except FileNotFoundError:
            print(f'File not found: {img_name}, {tar_name}')               

        # If image is smaller than patch size, just use as is, else crop patches
        if img.shape[0] <= self.image_size[0] and img.shape[1] <= self.image_size[1]:
            if self.pre_transform is not None:
                img, tar = self.pre_transform(img, tar)
            if self.transform is not None:
                img, tar = self.transform(img, tar)
            x = torch.from_numpy(img).type(self.inputs_dtype)
            y = torch.from_numpy(tar).type(self.targets_dtype)
            return x, y, img_name_str
        else:
            # Patch extraction logic for larger images
            # This is a sliding window approach with overlap
            patch_size = [self.image_size[0], self.image_size[1]]
            self.patch_batch = 0.75
            downsample_level = 0.5
            stride = [int(patch_size[0] * (1 - self.patch_batch) * downsample_level),
                      int(patch_size[1] * (1 - self.patch_batch) * downsample_level)]
            n_patches = [1 + floor((img.shape[0] - patch_size[0]) / stride[0]),
                         1 + floor((img.shape[1] - patch_size[1]) / stride[1])]
            start_coords = [0, 0]
            row_starts = [int(start_coords[0] + (i * stride[0])) for i in range(0, n_patches[0])]
            col_starts = [int(start_coords[1] + (i * stride[1])) for i in range(0, n_patches[1])]
            row_starts.append(int(img.shape[0] - patch_size[0]))
            col_starts.append(int(img.shape[1] - patch_size[1]))
            
            x_list, y_list, patch_id_list = [], [], []
            for r_s in row_starts:
                for c_s in col_starts:
                    new_img = img[r_s:r_s + patch_size[0], c_s:c_s + patch_size[1], :]
                    new_tar = np.zeros((new_img.shape[0], new_img.shape[1])) if tar_name is None else tar[r_s:r_s + patch_size[0], c_s:c_s + patch_size[1]]

                    if self.pre_transform is not None:
                        new_img, new_tar = self.pre_transform(new_img, new_tar)
                    if self.transform is not None:
                        x, y = self.transform(new_img, new_tar)
                    # Adding batch dimension
                    x = x[None,:,:,:]
                    y = y[None,:,:,:]
                    x = torch.from_numpy(x).type(self.inputs_dtype)
                    y = torch.from_numpy(y).type(self.targets_dtype)
                    patch_id = f"{img_name_str}_{r_s}_{c_s}"
                    x_list.append(x)
                    y_list.append(y)
                    patch_id_list.append(patch_id)
            return x_list, y_list, patch_id_list
    
    def add_sub_categories(self,sub_categories, sub_cat_column):
        # Implementing sub-category weighting if provided with dataframe
        sub_cat_counts = sub_categories[sub_cat_column].value_counts(normalize=True).to_dict()
        print(f'Dataset composition: {sub_cat_counts}')

        # Applying inverse weight to each sample (evens out sampling)
        sub_cat_counts = {i:1.0-j for i,j in zip(list(sub_cat_counts.keys()),list(sub_cat_counts.values()))}

        # Calculating sample weight
        self.sample_weight = []
        for s in self.cached_names:
            sample_name = s.split('/')[-1]
            print(sample_name)
            if sample_name in sub_categories['Image_Names'].tolist():
                sample_label = sub_categories[sub_categories['Image_Names'].str.match(sample_name)]['Labels'].tolist()[0]
                print(sample_label)
                self.sample_weight.append(sub_cat_counts[sample_label])
            else:
                print(f'sample not in dataframe: {sample_name}')
                self.sample_weight.append(0)

        self.sample_weight = [i/sum(self.sample_weight) for i in self.sample_weight]
        print(f'sample_weight: {self.sample_weight}')
    
    def __iter__(self):
        
        self.item_idx = -1

        return self
    
    def __next__(self):
        self.item_idx+=1
        return self[self.item_idx]

    def normalize_cache(self,means,stds):
        # Applying normalization to a dataset according to a given set of means and standard deviations per channel
        for img,tar in self.cached_data:
            img = np.float64(img)
            for j in range(img.shape[-1]):
                img[:,:,j] -= means[j]
                img[:,:,j] /= stds[j]

# This is a custom wrapper class that behaves like Subset and forwards attributes automatically used in KFold or train-test split
class AttrSubset(Dataset):
    def __init__(self, dataset, indices):
        self._dataset = dataset   # use private names to avoid conflicts
        self._indices = indices

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def __len__(self):
        return len(self._indices)

    def __getattr__(self, attr):
        # Forward anything not found on self to the underlying dataset
        return getattr(self._dataset, attr)

    def __setattr__(self, key, value):
        # If it's one of our private attributes, set it here
        if key in {'_dataset', '_indices'}:
            super().__setattr__(key, value)
        else:
            setattr(self._dataset, key, value)

        
def make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,parameters, ema=False):
    
    img_size = [int(i) for i in parameters['preprocessing']['image_size'].split(',')]
    mask_size = [int(i) for i in parameters['preprocessing']['mask_size'].split(',')]
    color_transform = parameters['preprocessing']['color_transform']

    if 'image_means' in parameters['preprocessing']:
        image_means = [float(i) for i in parameters['preprocessing']['image_means'].split(',')]
        image_stds = [float(i) for i in parameters['preprocessing']['image_stds'].split(',')]
    else:
        # Normalization is normally subtracting mean and dividing by standard deviation
        # Setting the image_means to 0 and stds to 1 nullifies the effect
        image_means = [float(0.0) for i in range(img_size[-1])]
        image_stds = [float(1.0) for i in range(img_size[-1])]

    if phase == 'train':

        pre_transforms = ComposeDouble([
                FunctionWrapperDouble(resize_special,
                                    input=True,
                                    target=False,
                                    output_size=img_size,
                                    transform=color_transform),
                FunctionWrapperDouble(resize,
                                    input=False,
                                    target=True,
                                    output_shape=mask_size),
                FunctionWrapperDouble(normalize,
                                      input=True,
                                      target=False,
                                      mean=image_means,
                                      std=image_stds)
        ])        

        # Continuous target type augmentations
        transforms_training = ComposeDouble([
            AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
            AlbuSeg2d(albumentations.RandomBrightnessContrast(p=0.2)),
            AlbuSeg2d(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5,rotate_limit=45,interpolation=1,p=0.1)),
            AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, input=True, target=True,source=-1,destination=0)
        ])
 
        transforms_validation = ComposeDouble([
            FunctionWrapperDouble(np.moveaxis,input=True,target=True,source=-1,destination=0)
        ])
        if not ema:
            dataset_train = SegmentationDataSet(inputs=train_img_paths,
                                             targets=train_tar,
                                             transform=transforms_training,
                                             pre_transform=pre_transforms,
                                             parameters=parameters)
            # For debugging:        
            from torch.utils.data import DataLoader
            # import matplotlib.pyplot as plt            
            from CollagenSegUtils import visualize_continuous
            
            train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, pin_memory=True)
            train_loader_iter = iter(train_loader)
            x, y, input_ID = next(train_loader_iter)
            print(f'x shape: {x.shape}')
            print(f'y shape: {y.shape}')            
            print(f'input_ID: {input_ID}')     
            print(f'x min: {x.min()}, x max: {x.max()}')
            # Create color images for source and target
            x = x[0] # remove batch dimension
            if x.shape[0] == 2:
                # DUET and Brightfield: Only G channels
                x_vis = np.stack((
                    np.zeros_like(x[0]),     # R for DUET
                    x[0].cpu().numpy(),      # G for DUET
                    np.zeros_like(x[0]),     # B for DUET
                    np.zeros_like(x[0]),     # R for Brightfield
                    x[1].cpu().numpy(),      # G for Brightfield
                    np.zeros_like(x[0])      # B for Brightfield
                ), axis=0)
            elif x.shape[0] == 6:
                # Full RGB for DUET and Brightfield
                x_vis = x.cpu().numpy()
            elif x.shape[0] == 4:
                # DUET has only green channel, Brightfield has RGB
                x_vis = np.stack((
                    np.zeros_like(x[0]),     # R for DUET
                    x[0].cpu().numpy(),      # G for DUET
                    np.zeros_like(x[0]),     # B for DUET
                    x[1].cpu().numpy(),      # R for Brightfield
                    x[2].cpu().numpy(),      # G for Brightfield
                    x[3].cpu().numpy()       # B for Brightfield
                ), axis=0)
            else:
                raise ValueError(f"Unexpected number of channels: {x.shape[0]}")

            img_dict = {
                'Source Images': np.concatenate((x_vis[:3, :, :], x_vis[3:, :, :]), axis=-2), 
                'Ground_Truth': y[0].cpu().numpy()
            }

            fig = visualize_continuous(img_dict, 'comparison')
            fig.savefig(f"{parameters['output_dir']}/Fisrt_Sample_source.png")
            
        else:
            dataset_train = SegmentationDataSet(inputs=train_img_paths,
                                                targets=train_tar,
                                                transform=transforms_validation,
                                                pre_transform=pre_transforms,
                                                parameters=parameters)                
        
        dataset_valid = SegmentationDataSet(inputs=valid_img_paths,
                                             targets=valid_tar,
                                             transform=transforms_validation,
                                             pre_transform=pre_transforms,
                                             parameters=parameters)
        
    elif phase == 'test':

        pre_transforms = ComposeDouble([
                FunctionWrapperDouble(resize_special,
                                    input=True,
                                    target=False,
                                    output_size=img_size,
                                    transform=color_transform),
                FunctionWrapperDouble(resize,
                                    input=False,
                                    target=True,
                                    output_shape=mask_size),
                FunctionWrapperDouble(normalize,
                                        input=True,
                                        target=False,
                                        mean=image_means,
                                        std=image_stds)
            ])
        
        transforms_testing = ComposeDouble([
                FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0),
                ])

        # this is 'None' because we are just testing the network
        dataset_train = None
        
        dataset_valid = SegmentationDataSet(inputs=valid_img_paths,
                                            targets=valid_tar,
                                            transform=transforms_testing,
                                            pre_transform=pre_transforms,
                                            parameters=parameters)


    return dataset_train, dataset_valid
