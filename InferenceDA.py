# -*- coding: utf-8 -*-
"""
CS Testing - Predictions Only
Saves predictions per fold and target dataset
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from Models import create_model
from Cluster import Clusterer


def Test_Network(model_path, dataset_valid, nept_run, test_parameters):
    """
    Run predictions with a trained model and save output masks.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_details = test_parameters['model_details']
    model = create_model(model_details, n_classes=1)
    
    state_dict = torch.load(model_path, map_location=device)
    if 'combine_layers.weight' in state_dict:
        state_dict['segmentation_head.weight'] = state_dict.pop('combine_layers.weight')
    if 'combine_layers.bias' in state_dict:
        state_dict['segmentation_head.bias'] = state_dict.pop('combine_layers.bias')
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    clusterer = Clusterer(test_parameters)
    
    output_dir = test_parameters['current_target_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    datasets_valid = dataset_valid if isinstance(dataset_valid, list) else [dataset_valid]

    for d_idx, ds in enumerate(datasets_valid):
        print(f"\n--- Predicting dataset {d_idx+1}/{len(datasets_valid)} ---")
        
        with tqdm(range(len(ds)), desc=f'Predicting (Dataset {d_idx+1})') as pbar:
            for i in range(len(ds)):
                sample = ds[i]
                
                # Patch-based prediction
                if ds.patch_batch:
                    image_list, _, patch_id_list = sample
                    original_image_size = ds.original_size
                    final_pred_mask = np.zeros(original_image_size, dtype=np.float32)
                    overlap_mask = np.zeros_like(final_pred_mask)
                    
                    patch_size = [int(x) for x in test_parameters['preprocessing']['image_size'].split(',')[:-1]]
                    image_name = os.path.basename(ds.inputs[i])
                    save_name = image_name.replace('.' + image_name.split('.')[-1], '_prediction.tif')
                    
                    for image, patch_id in zip(image_list, patch_id_list):
                        pred_mask = model(image.to(device))
                        pred_mask_np = pred_mask.detach().cpu().numpy()
                        if pred_mask_np.ndim == 4:
                            pred_mask_np = pred_mask_np[0,0,:,:]
                        pred_mask_np = cv2.normalize(pred_mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        
                        row_start = int(patch_id.split('_')[-2])
                        col_start = int(patch_id.split('_')[-1])
                        final_pred_mask[row_start:row_start+patch_size[0], col_start:col_start+patch_size[1]] += pred_mask_np
                        overlap_mask[row_start:row_start+patch_size[0], col_start:col_start+patch_size[1]] += 1
                    
                    final_pred_mask = np.divide(final_pred_mask, np.maximum(overlap_mask, 1))
                    final_pred_mask = final_pred_mask.astype(np.uint8)
                    
                    im = Image.fromarray(final_pred_mask)
                    im.save(os.path.join(output_dir, save_name))
                    pbar.update(1)
                
                # Small-image prediction
                else:
                    image, _, input_name = sample
                    input_name = os.path.basename(input_name)
                    image = image.unsqueeze(0)
                    
                    pred_mask = model(image.to(device))
                    pred_mask_np = pred_mask.detach().cpu().numpy()
                    if pred_mask_np.ndim == 4:
                        pred_mask_np = pred_mask_np[0,0,:,:]
                    pred_mask_np = cv2.normalize(pred_mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    save_name = input_name.replace('.' + input_name.split('.')[-1], '_prediction.tif')
                    im = Image.fromarray(pred_mask_np)
                    im.save(os.path.join(output_dir, save_name))
                    pbar.update(1)
