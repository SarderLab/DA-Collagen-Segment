# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:28:08 2025

@author: fafsari


CS Testing best model

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb


"""

import torch
from tqdm import tqdm
import os
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd

import neptune

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from Utils import visualize_continuous, get_metrics
from Cluster import Clusterer
from Models import create_model
    
        
def Test_Network(model_path, dataset_valid, nept_run, test_parameters):    
    
    model_details = test_parameters['model_details']

    if 'scaler_means' not in model_details:
        test_parameters['model_details']['scaler_means'] = None
        
    encoder = model_details['encoder']
    encoder_weights = model_details['encoder_weights']

    # Loading clusterer to cluster latent features
    clusterer = Clusterer(test_parameters)

    ann_classes = model_details['ann_classes']
    active = model_details['active']
    target_type = model_details['target_type']
    output_dir = test_parameters['output_dir']

    if active == 'None':
        active = None

    if target_type=='binary':
        # n_classes = len(ann_classes)
        n_classes = 1
    elif target_type == 'nonbinary':
        n_classes = 1

    in_channels = int(test_parameters['preprocessing']['image_size'].split(',')[-1])
    output_type = 'prediction'

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    model = create_model(model_details, n_classes)

    state_dict =torch.load(model_path, map_location=device)
        
    # Update the state dict: rename 'combine_layers' to 'segmentation_head'
    if 'combine_layers.weight' in state_dict:
        state_dict['segmentation_head.weight'] = state_dict.pop('combine_layers.weight')
    if 'combine_layers.bias' in state_dict:
        state_dict['segmentation_head.bias'] = state_dict.pop('combine_layers.bias')
    
    # Load the updated state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # Save the modified model
    torch.save(model.state_dict(), model_path)
    print(f"Loading model from {model_path}")    
        
    print("model successfully loaded!")
    
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Evaluating model on test set
        #test_dataloader = DataLoader(dataset_valid)
        
        if "Source" in test_parameters['mode']:
            test_output_dir = output_dir+'/Source/Testing_Output/'
        elif "Target" in test_parameters['mode']:
            test_output_dir = output_dir+'/Target/Testing_Output/'
        else:
            test_output_dir = output_dir+'/Testing_Output/'
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)

        # Normalize into a list for unified handling
        datasets_valid = dataset_valid if isinstance(dataset_valid, list) else [dataset_valid]

        # Check if any dataset requires testing metrics
        if any(ds.testing_metrics for ds in datasets_valid):
            print('Calculating testing metrics with target type:', target_type)

            if target_type == 'binary':
                metrics_calculator = BinaryMetrics()
                testing_metrics_df = pd.DataFrame(
                    columns=['Dice', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'AUC', 'MSE', 'R2']
                )
            elif target_type == 'nonbinary':
                metrics_calculator = []  # (or maybe a proper NonBinaryMetrics class if you make one later)
                testing_metrics_df = pd.DataFrame(columns=['MSE', 'Norm_MSE'])
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
        else:
            metrics_calculator = None
            testing_metrics_df = None
        # print('Testing metrics dataframe initialized:', testing_metrics_df.columns, index=None)
                
        # Setting up iterator(s) to generate images from the validation dataset for prediction
        # if isinstance(dataset_valid, list):
        #     data_iterators = [iter(ds) for ds in dataset_valid]
        # else:
        #     data_iterators = iter(dataset_valid)

        all_latent_features = None
        clustering_labels = []

        if isinstance(dataset_valid, list):
            if any(ds.patch_batch for ds in dataset_valid):
                print('Using patch prediction pipeline (at least one target requires patches)')
            else:
                print('Images are the same size as the model inputs (all targets)')
        else:
            if dataset_valid.patch_batch:
                print('Using patch prediction pipeline')
            else:
                print('Images are the same size as the model inputs')

        # --- Prediction Loop ---
        for d_idx, ds in enumerate(datasets_valid):
            print(f"\n=== Running predictions for dataset {d_idx+1}/{len(datasets_valid)} ===")
            
            with tqdm(range(len(ds)), desc=f'Predicting (Dataset {d_idx+1})') as pbar:
                for i in range(len(ds)):
                    sample = ds[i]

                    # ---------------- Patch Prediction Pipeline for large images ----------------
                    if ds.patch_batch:  #isinstance(sample[0], list):  
                        # Large image → patch lists
                        image_list, _, patch_id_list = sample

                        # Original image size from first patch (reconstructing)
                        # safer way would be reading again from inputs[i], but here patch stitching is enough
                        # original_image = imread(str(ds.inputs[i]))
                        original_image_size = ds.original_size  #np.shape(original_image)
                        final_pred_mask = np.zeros((original_image_size[0], original_image_size[1]))
                        overlap_mask = np.zeros_like(final_pred_mask)

                        patch_size = [int(x) for x in test_parameters['preprocessing']['image_size'].split(',')[0:-1]]
                        image_name = os.path.basename(ds.inputs[i])
                        save_name = image_name.replace('.' + image_name.split('.')[-1], '_prediction.tif')

                        for image, patch_id in zip(image_list, patch_id_list):
                            pred_mask = model(image.to(device))

                            # Latent features (optional)
                            if test_parameters['model_details']['scaler_means'] is not None:
                                pred_latent_features = clusterer.cluster_in_loop(model, image.to(device))
                                pred_latent_features = pred_latent_features.cpu().numpy()
                                if all_latent_features is None:
                                    all_latent_features = pred_latent_features
                                else:
                                    all_latent_features = np.concatenate((all_latent_features, pred_latent_features), axis=0)
                                clustering_labels.append({'Full_Image_Name': image_name, 'Patch_Name': patch_id})

                            # Prediction array
                            pred_mask_img = pred_mask.detach().cpu().numpy()

                            # Get patch coordinates from patch_id
                            row_start = int(patch_id.split('_')[-2])
                            col_start = int(patch_id.split('_')[-1])

                            final_pred_mask[row_start:row_start+patch_size[0], col_start:col_start+patch_size[1]] += np.squeeze(pred_mask_img*255)
                            overlap_mask[row_start:row_start+patch_size[0], col_start:col_start+patch_size[1]] += 1

                            pbar.update(1)

                        # Normalize overlaps
                        final_pred_mask = np.multiply(final_pred_mask, 1/np.maximum(overlap_mask, 1))

                        # Save prediction
                        im = Image.fromarray(final_pred_mask.astype(np.uint8))
                        im = im.filter(ImageFilter.SMOOTH_MORE)
                        im.save(os.path.join(test_output_dir, save_name))

                    # ---------------- Image Prediction for Small Images ----------------
                    else:
                        image, target, input_name = sample
                        input_name = os.path.basename(input_name)

                        # Model prediction
                        image = image[None, :, :, :]
                        pred_mask = model(image.to(device))

                        # Latent feature clustering
                        if test_parameters['model_details']['scaler_means'] is not None:
                            pred_latent_features = clusterer.cluster_in_loop(model, image.to(device))
                            pred_latent_features = pred_latent_features.cpu().numpy()
                            if all_latent_features is None:
                                all_latent_features = pred_latent_features
                            else:
                                all_latent_features = np.concatenate((all_latent_features, pred_latent_features), axis=0)
                            clustering_labels.append({'Patch_Name': input_name})

                        # Convert outputs
                        pred_mask_img = pred_mask.detach().cpu().numpy()
                        target_img = target.cpu().numpy()

                        # Metrics calculation
                        if ds.testing_metrics:
                            testing_metrics_df = pd.concat([
                                testing_metrics_df,
                                pd.DataFrame(get_metrics(pred_mask.detach().cpu(), target.cpu(), input_name, metrics_calculator, target_type))
                            ], ignore_index=True)

                        # Reformat input image depending on channels
                        image_np = image.cpu().numpy()
                        if isinstance(in_channels, int):
                            if in_channels == 6:
                                image_np = np.concatenate((image_np[:, 0:3, :, :], image_np[:, 3:6, :, :]), axis=2)
                            elif in_channels == 4:
                                image_np = np.concatenate((np.stack((image_np[:, 0, :, :],)*3, axis=1), image_np[:, 1:4, :, :]), axis=2)
                            elif in_channels == 2:
                                image_np = np.concatenate((image_np[:, 0, :, :], image_np[:, 1, :, :]), axis=-1)
                        elif isinstance(in_channels, list):
                            if sum(in_channels) == 6:
                                image_np = np.concatenate((image_np[:, 0:3, :, :], image_np[:, 3:6, :, :]), axis=2)
                            elif sum(in_channels) == 2:
                                image_np = np.concatenate((image_np[:, 0, :, :][None, :, :], image_np[:, 1, :, :][None, :, :]), axis=2)

                        # Visualization
                        img_dict = {'Image': image_np, 'Pred_Mask': pred_mask_img, 'Ground_Truth': target_img}
                        fig = visualize_continuous(img_dict, output_type)

                        pbar.update(1)

                        # Save results
                        if output_type == 'comparison':
                            fig.savefig(os.path.join(test_output_dir, f'Test_Example_{input_name}'))
                        elif output_type == 'prediction':
                            im = Image.fromarray((fig*255).astype(np.uint8))
                            ext = input_name.split('.')[-1]
                            im.save(os.path.join(test_output_dir, f'Test_Example_{input_name.replace(ext,'tif')}'))
        
            # Save metrics (per dataset or per k-fold)
            if 'current_k_fold' in test_parameters:
                testing_metrics_df.to_excel(os.path.join(test_output_dir, f'Test_Metrics_fold_{test_parameters["current_k_fold"]}_ds{d_idx+1}.xlsx'), index=False)
            else:
                testing_metrics_df.to_excel(os.path.join(test_output_dir, f'Test_Metrics_ds{d_idx+1}.xlsx'), index=False)
               
            

