# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:39:49 2024

@author: fafsari

This is the patching script for Deep-DA-DUET

"""

from PIL import Image
import numpy as np
import os
from math import floor

def is_blank_patch(img, threshold=25):
    """Optional: define if an image is blank based on pixel stats."""
    return np.array(img).std() < threshold

def patch_image(image_files, patch_dirs, case_ID, image_size=(512, 512)):
    # Load all images
    imgs = [Image.open(image_files[i]) for i in range(len(image_files))]

    # Use the first image to determine patching grid
    ref_img = imgs[0]
    patch_size = (int(image_size[0]), int(image_size[1]))
    patch_batch = 0.75
    downsample_level = 0.5
    stride = [
        int(patch_size[0] * (1 - patch_batch) * downsample_level),
        int(patch_size[1] * (1 - patch_batch) * downsample_level)
    ]

    n_patches = [
        1 + floor((ref_img.size[1] - patch_size[0]) / stride[0]),  # Height = size[1]
        1 + floor((ref_img.size[0] - patch_size[1]) / stride[1])   # Width = size[0]
    ]
    start_coords = [0, 0]

    row_starts = [int(start_coords[0] + (i * stride[0])) for i in range(n_patches[0])]
    col_starts = [int(start_coords[1] + (i * stride[1])) for i in range(n_patches[1])]
    row_starts.append(ref_img.size[1] - patch_size[0])
    col_starts.append(ref_img.size[0] - patch_size[1])
    
    print(f"Number of patches (rows, cols): {len(row_starts)}, {len(col_starts)}, Total: {len(row_starts) * len(col_starts)}")

    # Check for blank images and create patch directories
    for img, image_name, patch_dir in zip(imgs, image_files, patch_dirs):
        print("patch directory:", patch_dir)
        if not os.path.isdir(patch_dir):
            print(f"Creating patch directory: {patch_dir}")
            os.makedirs(patch_dir, exist_ok=True)
        else:
            print(f"Patch directory already exists! {len(os.listdir(patch_dir))} files found.")

    # For each patch coordinate, crop and save patches from all 3 images
    for r_s in row_starts:
        for c_s in col_starts:
            box = (c_s, r_s, c_s + patch_size[0], r_s + patch_size[1])
            
            # Crop the patch from reference image (first in the list)
            ref_patch = imgs[0].crop(box)

            if is_blank_patch(ref_patch):
                continue  # Skip this location for all images

            for img, image_name, patch_dir in zip(imgs, image_files, patch_dirs):
                extension = os.path.splitext(image_name)[-1]
                base_name = os.path.basename(image_name).replace(extension, '')
                patch_name = f"{base_name}_{r_s}_{c_s}{extension}"

                patch = img.crop(box)
                patch.save(os.path.join(patch_dir, patch_name))


# Base path to the folder containing subfolders
base_path = "/path/to/image folder"
patch_path = "/path/to/patch folder"

case_ID = "case_001"  # Example case ID, can be set dynamically
# Names of the subfolders
subfolders = ["B", "F", "C"]


# List of image file paths from all subfolders
B_image_files = [os.path.join(base_path, subfolders[0], f) 
                 for f in os.listdir(os.path.join(base_path, subfolders[0])) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
                ]
F_image_files = [os.path.join(base_path, subfolders[1], f) 
                 for f in os.listdir(os.path.join(base_path, subfolders[1])) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
                ]
C_image_files = [os.path.join(base_path, subfolders[2], f) 
                 for f in os.listdir(os.path.join(base_path, subfolders[2])) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
                ]

# for b_image in B_image_files: #zip(all_image_files[0], all_image_files[1], all_image_files[2]):
for b_image, f_image, c_image in zip(B_image_files, F_image_files, C_image_files):
    # Ensure the images are in the correct format and have the same basename        
    image_basename = os.path.basename(b_image)
    print(f"Processing image: {image_basename}")
        
    ext = os.path.splitext(image_basename)[-1]
    
    all_images = [b_image, f_image, c_image]
        
    p_dirs = [os.path.join(patch_path, subfolders[i]) for i in range(len(subfolders))]
    
    patch_image(all_images, p_dirs, case_ID)
    
