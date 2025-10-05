# -*- coding: utf-8 -*-
"""
CS Testing best model with dynamic DUET green channel selection, robust patch merging,
and saving predictions as JPG with original base filenames.
Automatically selects fluorescent green channel based on preprocessing transform.
"""

import torch
from tqdm import tqdm
import os
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import cv2

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from Utils import visualize_continuous, get_metrics
from Cluster import Clusterer
from Models import create_model


def Test_Network(model_path, dataset_valid, nept_run, test_parameters):    
    model_details = test_parameters['model_details']

    if 'scaler_means' not in model_details:
        test_parameters['model_details']['scaler_means'] = None
       
    clusterer = Clusterer(test_parameters)
   
    active = model_details['active']
    target_type = model_details['target_type']
    output_dir = test_parameters['output_dir']

    if active == 'None':
        active = None

    n_classes = 1  # binary or nonbinary
    in_channels = int(test_parameters['preprocessing']['image_size'].split(',')[-1])
    output_type = 'prediction'

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    model = create_model(model_details, n_classes)
    state_dict = torch.load(model_path, map_location=device)
        
    # Rename 'combine_layers' to 'segmentation_head' if present
    if 'combine_layers.weight' in state_dict:
        state_dict['segmentation_head.weight'] = state_dict.pop('combine_layers.weight')
    if 'combine_layers.bias' in state_dict:
        state_dict['segmentation_head.bias'] = state_dict.pop('combine_layers.bias')
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    torch.save(model.state_dict(), model_path)
    print(f"Loading model from {model_path}")    
    print("Model successfully loaded!")
    
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Determine test output directory
        if "Source" in test_parameters['mode']:
            test_output_dir = os.path.join(output_dir,'Source','Testing_Output')
        elif "Target" in test_parameters['mode']:
            test_output_dir = os.path.join(output_dir,'Target','Testing_Output')
        else:
            test_output_dir = os.path.join(output_dir,'Testing_Output')
        os.makedirs(test_output_dir, exist_ok=True)

        datasets_valid = dataset_valid if isinstance(dataset_valid, list) else [dataset_valid]

        # Initialize metrics
        if any(ds.testing_metrics for ds in datasets_valid):
            if target_type == 'binary':
                metrics_calculator = BinaryMetrics()
                testing_metrics_df = pd.DataFrame(
                    columns=['Dice', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'AUC', 'MSE', 'R2', 'ImgLabel']
                )
            else:
                metrics_calculator = []
                testing_metrics_df = pd.DataFrame(columns=['MSE', 'Norm_MSE', 'ImgLabel'])
        else:
            metrics_calculator = None
            testing_metrics_df = None                

        all_latent_features = None
        clustering_labels = []

        for d_idx, ds in enumerate(datasets_valid):
            print(f"\n=== Running predictions for dataset {d_idx+1}/{len(datasets_valid)} ===")
            
            ds_pred_dir = os.path.join(test_output_dir, f'ds{d_idx+1}')
            os.makedirs(ds_pred_dir, exist_ok=True)

            transform_type = getattr(ds, 'transform', None)

            with tqdm(range(len(ds)), desc=f'Predicting (Dataset {d_idx+1})') as pbar:
                for i in range(len(ds)):
                    sample = ds[i]

                    # ---------------- Patch Prediction ----------------
                    if ds.patch_batch:  
                        image_list, _, patch_id_list = sample
                        original_image_size = ds.original_size
                        final_pred_mask = np.zeros((original_image_size[0], original_image_size[1]))
                        overlap_mask = np.zeros_like(final_pred_mask)

                        patch_size = [int(x) for x in test_parameters['preprocessing']['image_size'].split(',')[0:-1]]
                        image_name = os.path.basename(ds.inputs[i])
                        base_name = os.path.splitext(image_name)[0]
                        save_name = base_name + ".jpg"  # save as JPG

                        for image, patch_id in zip(image_list, patch_id_list):
                            pred_mask = model(image.to(device))

                            # --- Process green channel × binary ---
                            pred_mask_np = pred_mask.detach().cpu().numpy()
                            if pred_mask_np.ndim == 4:
                                pred_mask_np = pred_mask_np[0, 0, :, :]
                            pred_mask_np = cv2.normalize(pred_mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                            img_F = image[0].cpu().numpy().transpose(1, 2, 0)

                            # --- Select correct fluorescent green channel ---
                            if transform_type in ['multi_input_green', 'multi_input_mean', 'multi_input_green_invbf', 'multi_input_green_rgb']:
                                green_F = img_F[:, :, 0]
                            elif transform_type == 'multi_input_rgb':
                                green_F = img_F[:, :, 1]
                            else:
                                green_F = img_F[:, :, 0]  # default fallback

                            green_F_uint8 = (green_F * 255).astype(np.uint8)
                            _, binary_F = cv2.threshold(green_F_uint8, 70, 255, cv2.THRESH_BINARY)

                            if binary_F.shape != pred_mask_np.shape:
                                binary_F = cv2.resize(binary_F, (pred_mask_np.shape[1], pred_mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                            multiplied = cv2.multiply(binary_F, pred_mask_np)
                            multiplied_norm = cv2.normalize(multiplied, None, 0, 255, cv2.NORM_MINMAX)
                            blurred = cv2.GaussianBlur(multiplied_norm, (5, 5), 0)

                            # Latent features
                            if test_parameters['model_details']['scaler_means'] is not None:
                                pred_latent_features = clusterer.cluster_in_loop(model, image.to(device))
                                pred_latent_features = pred_latent_features.cpu().numpy()
                                if all_latent_features is None:
                                    all_latent_features = pred_latent_features
                                else:
                                    all_latent_features = np.concatenate((all_latent_features, pred_latent_features), axis=0)
                                clustering_labels.append({'Full_Image_Name': image_name, 'Patch_Name': patch_id})

                            row_start = int(patch_id.split('_')[-2])
                            col_start = int(patch_id.split('_')[-1])
                            final_pred_mask[row_start:row_start+patch_size[0], col_start:col_start+patch_size[1]] += blurred
                            overlap_mask[row_start:row_start+patch_size[0], col_start:col_start+patch_size[1]] += 1
                            pbar.update(1)

                        # Normalize by overlaps and save
                        final_pred_mask = np.multiply(final_pred_mask, 1/np.maximum(overlap_mask, 1))
                        im = Image.fromarray(final_pred_mask.astype(np.uint8))
                        im = im.filter(ImageFilter.SMOOTH_MORE)
                        im.save(os.path.join(ds_pred_dir, save_name))  # save as JPG

                    # ---------------- Small Image Prediction ----------------
                    else:
                        image, target, input_name = sample
                        input_name = os.path.basename(input_name)
                        base_name = os.path.splitext(input_name)[0]
                        save_name = base_name + ".jpg"  # save as JPG

                        image = image[None, :, :, :]
                        pred_mask = model(image.to(device))

                        # --- Process green channel × binary ---
                        pred_mask_np = pred_mask.detach().cpu().numpy()
                        if pred_mask_np.ndim == 4:
                            pred_mask_np = pred_mask_np[0, 0, :, :]
                        pred_mask_np = cv2.normalize(pred_mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                        img_F = image[0].cpu().numpy().transpose(1, 2, 0)

                        # --- Select correct fluorescent green channel ---
                        if transform_type in ['multi_input_green', 'multi_input_mean', 'multi_input_green_invbf', 'multi_input_green_rgb']:
                            green_F = img_F[:, :, 0]
                        elif transform_type == 'multi_input_rgb':
                            green_F = img_F[:, :, 1]
                        else:
                            green_F = img_F[:, :, 0]  # default fallback

                        green_F_uint8 = (green_F * 255).astype(np.uint8)
                        _, binary_F = cv2.threshold(green_F_uint8, 70, 255, cv2.THRESH_BINARY)

                        if binary_F.shape != pred_mask_np.shape:
                            binary_F = cv2.resize(binary_F, (pred_mask_np.shape[1], pred_mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                        multiplied = cv2.multiply(binary_F, pred_mask_np)
                        multiplied_norm = cv2.normalize(multiplied, None, 0, 255, cv2.NORM_MINMAX)
                        blurred = cv2.GaussianBlur(multiplied_norm, (5, 5), 0)

                        # Latent features
                        if test_parameters['model_details']['scaler_means'] is not None:
                            pred_latent_features = clusterer.cluster_in_loop(model, image.to(device))
                            pred_latent_features = pred_latent_features.cpu().numpy()
                            if all_latent_features is None:
                                all_latent_features = pred_latent_features
                            else:
                                all_latent_features = np.concatenate((all_latent_features, pred_latent_features), axis=0)
                            clustering_labels.append({'Patch_Name': input_name})

                        target_img = target.cpu().numpy()
                        pred_mask_tensor = torch.tensor(blurred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        blurred_vis = np.expand_dims(blurred, axis=-1)

                        # Metrics
                        if ds.testing_metrics:
                            testing_metrics_df = pd.concat([
                                testing_metrics_df,
                                pd.DataFrame(get_metrics(pred_mask_tensor, target.cpu(), input_name, metrics_calculator, target_type))
                            ], ignore_index=True)

                        img_dict = {'Image': img_F, 'Pred_Mask': blurred_vis, 'Ground_Truth': target_img}
                        fig = visualize_continuous(img_dict, output_type)
                        pbar.update(1)

                        if output_type == 'comparison':
                            fig.savefig(os.path.join(ds_pred_dir, save_name))
                        elif output_type == 'prediction':
                            im = Image.fromarray(blurred.astype(np.uint8))
                            im.save(os.path.join(ds_pred_dir, save_name))  # save as JPG

            # Save metrics per dataset
            if testing_metrics_df is not None:
                if 'current_k_fold' in test_parameters:
                    metrics_file = os.path.join(test_output_dir, f'Test_Metrics_fold_{test_parameters["current_k_fold"]}_ds{d_idx+1}.xlsx')
                else:
                    metrics_file = os.path.join(test_output_dir, f'Test_Metrics_ds{d_idx+1}.xlsx')
                testing_metrics_df.to_excel(metrics_file, index=False)

    print("Testing completed!")
