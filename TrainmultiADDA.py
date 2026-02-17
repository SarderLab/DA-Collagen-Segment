# -*- coding: utf-8 -*-
"""

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

A lot of different segmentation networks are available from the 'segmentation_models_pytorch' module
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import albumentations

from albumentations import Affine

from Augmentation_Functions import *
from Models import create_model
from Loss import *
from Utils import visualize_continuous

from PIL import Image
import pandas as pd
from tqdm import tqdm
import sys
import os
import time
import gc



# Calculate MMD between source and target features
def compute_mmd(source_features, target_features):
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)
    mmd_loss = torch.norm(source_mean - target_mean, p=2)
    return mmd_loss

def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_target_variables(source_model, target_model, alpha, global_step):
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    p = 0.5
    alpha = min(1 - 1 / (global_step + 1)**p, alpha)
    # alpha = 0.6
    one_minus_alpha = 1.0 - alpha    
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):    
        # target_param.data.mul_(alpha).add_(one_minus_alpha, source_param.data)
        target_param = alpha * target_param.data + one_minus_alpha * source_param.data

def linked_batch_augmentation(input_batch, preds_unsup):
    
    teacher_transforms = ComposeDouble([
            AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
            AlbuSeg2d(albumentations.RandomBrightnessContrast(p=0.2)),
            AlbuSeg2d(
        Affine(
            translate_percent={"x": 0.0625, "y": 0.0625},  # same as shift_limit
            scale=(0.5, 1.5),  # scale_limit=0.5 → 0.5–1.5
            rotate=(-45, 45),  # same as rotate_limit
            interpolation=1,
            p=0.1
        )
    ),
            AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0)
        ])

    input_batch_size = input_batch.size(0)

    input_batch_cpu = input_batch.cpu().detach().numpy()
    preds_unsup_cpu = preds_unsup.cpu().detach().numpy()

    samples_linked_aug = []
    for sample_idx in range(input_batch_size):
        x = np.moveaxis(input_batch_cpu[sample_idx], 0, -1)  # Move channels to last dimension
        y = np.moveaxis(preds_unsup_cpu[sample_idx], 0, -1)  # Move channels to last dimension

        # Apply transformations
        augmented_x, augmented_y = teacher_transforms(x, y)

        samples_linked_aug.append((augmented_x, augmented_y))
            
    # Separate the augmented samples into inputs and predictions
    augmented_inputs, augmented_preds = zip(*samples_linked_aug)

    # Convert lists to numpy arrays
    augmented_inputs = np.array(augmented_inputs)
    augmented_preds = np.array(augmented_preds)

    # Convert numpy arrays back to PyTorch tensors
    augmented_inputs_tensor = torch.tensor(augmented_inputs, dtype=torch.float32)
    augmented_preds_tensor = torch.tensor(augmented_preds, dtype=torch.float32)

    return augmented_inputs_tensor, augmented_preds_tensor

def TrainingADDA_Loop(model_path, dataset_train_s, dataset_valid_s, dataset_train_t, dataset_valid_t, train_parameters, nept_run):
    
    model_details = train_parameters['model_details']

    if not model_details['architecture'] == 'DUnet':
        encoder = model_details['encoder']
        encoder_weights = model_details['encoder_weights']
        dropout_rate = model_details['dropout_rate']
        pretrained = True if model_details['pretrained']==1 else False
        nept_run['encoder'] = encoder
        nept_run['encoder_pre_train'] = encoder_weights

    output_type = 'comparison'
    active = model_details['active']
    target_type = model_details['target_type']
    ann_classes = model_details['ann_classes'].split(',')
    output_dir = train_parameters['output_dir']
    model_dir = output_dir+'/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if active=='None':
        active = None

    in_channels = int(model_details['in_channels'])
    
    if target_type=='binary':
        loss = smp.losses.DiceLoss(mode='binary')        
        #n_classes = len(ann_classes)
        n_classes = 1

    elif target_type=='nonbinary':
        if train_parameters['loss']=='MSE':
            loss = torch.nn.MSELoss(reduction='mean')
        elif train_parameters['loss'] == 'L1':
            loss = torch.nn.L1Loss(reduction='mean')
        elif train_parameters['loss'] == 'BCE':
            loss = torch.nn.BCELoss()
        n_classes = 1
    
    # image_size = train_parameters["preprocessing"]["image_size"]
    image_size = [int(i) for i in train_parameters['preprocessing']['image_size'].split(',')]
    n_channels = image_size[-1]//2
    image_size = (image_size[0], image_size[1])
    
    batch_size = int(train_parameters['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    

    nept_run['Architecture'] = model_details['architecture']
    nept_run['Loss'] = train_parameters['loss']
    nept_run['output_type'] = output_type
    nept_run['lr'] = train_parameters['lr']
    
    model = create_model(model_details, n_classes)
    # target_model = create_model(model_details, n_classes)
    
    model.to(device)
    # target_model.to(device)
    
    if pretrained:
        print(f"Loading pre-trained model from {model_path}")            
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)        

        # Update target model's state_dict with source model's parameters
        source_state_dict = model.state_dict()
        # target_state_dict = target_model.state_dict()
        # target_state_dict.update(source_state_dict)

        # Load the updated state_dict into target model
        # target_model.load_state_dict(target_state_dict, strict=False)

    # dummy_input = torch.zeros(batch_size, in_channels, image_size[0], image_size[1]).to(device)
    # _ = target_model(dummy_input)
    # for param in target_model.parameters():
    #     param.detach_()
    
    torch.cuda.empty_cache()
    
    # Sending loss to current device ('cuda','cuda:0','cuda:1',or 'cpu')    
    loss = loss.to(device) 
                
    # optimizer = torch.optim.Adam([
    #         dict(params = model.parameters(), lr = train_parameters['lr'], weight_decay = 0.0001)
    #         ])
    # -----------------------------------------------------
    # Create optimizers
    # -----------------------------------------------------
    optimizer_seg = torch.optim.Adam(model.Gs.parameters(), lr=train_parameters['lr'], weight_decay=0.0001)
    optimizer_disc = torch.optim.Adam(model.D.parameters(), lr=train_parameters['lr'], weight_decay=0.0001)
    optimizer_gen_t = torch.optim.Adam(model.Gt.parameters(), lr=train_parameters['lr'], weight_decay=0.0001)

    # Define the learning rate scheduler with adjusted patience    
    lr_plateau_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seg, mode='min', factor=0.1, patience=250)
    lr_plateau_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, mode='min', factor=0.1, patience=250)
    lr_plateau_gen_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen_t, mode='min', factor=0.1, patience=250)
        
    # Create data loader for the source
    train_loader_s = DataLoader(dataset_train_s, batch_size=batch_size, shuffle=True, pin_memory=True)    
    valid_loader_s = DataLoader(dataset_valid_s, batch_size=64, shuffle=True, pin_memory=True)
    train_iters_s = iter(train_loader_s)
    valid_iters_s = iter(valid_loader_s)
    
    # Create data loader for the target(s)
    if isinstance(dataset_train_t, list):
        # ---- Multi-target setup ----
        train_loaders_t = [
            DataLoader(dataset_train_t[i], batch_size=batch_size, shuffle=True, pin_memory=True) for i in range(len(dataset_train_t))
        ]
        valid_loaders_t = [
        DataLoader(dataset_valid_t[i], batch_size=64, shuffle=True, pin_memory=True) for i in range(len(dataset_valid_t))
        ]
    else:
        # ---- Single-target setup ----
        train_loaders_t = [DataLoader(dataset_train_t, batch_size=batch_size, shuffle=True, pin_memory=True)]
        valid_loaders_t = [DataLoader(dataset_valid_t, batch_size=64, shuffle=True, pin_memory=True)]
    
    train_iters_t = [iter(loader) for loader in train_loaders_t]
    valid_iters_t = [iter(loader) for loader in valid_loaders_t]
        
        
    # Maximum number of epochs defined here as well as how many steps between model saves and example outputs
    step_num = train_parameters['step_num']
    save_step = train_parameters['save_step']

    train_loss = 0
    seg_loss = 0
    val_loss_s = 0
    val_loss_t = 0
    best_val_loss = float('inf')

    # Recording training and validation loss
    train_loss_list = []
    train_loss_list_seg = []
    val_loss_list_s = []
    val_loss_list_t = []
    
           
    # for global_step in tqdm(range(1, step_num + 1), desc="Epochs"):
    with tqdm(total = step_num, position = 0, leave = True, file = sys.stdout) as pbar:
        start_time = time.time()
        
        global_step = 0
        
        while global_step < step_num:
            
            # Controlling the progress bar, printing training and validation losses
            if global_step == 1: 
                pbar.set_description(f'Step: {global_step}/{step_num}')
                pbar.update(global_step)
            elif global_step % 5 == 0:
                pbar.set_description(
                f"Step: {global_step}/{step_num}, "
                f"Train Loss: {train_loss:.4f} / "
                f"Val Loss Source: {val_loss_s:.4f} / "
                f"Val Loss Target: {val_loss_t:.4f}"
                )
                pbar.update(5)

            # ----------------------------
            # Get next SOURCE batch
            # ----------------------------
            try:
                source_train_batch = next(train_iters_s)
            except StopIteration:
                train_iters_s = iter(train_loader_s)
                source_train_batch = next(train_iters_s)

            source_train_inputs, source_train_masks, _ = source_train_batch
            source_train_inputs = source_train_inputs.to(device)
            source_train_masks = source_train_masks.to(device)

            # ----------------------------
            # Get next TARGET batch
            # ----------------------------
            for idx, t_iter in enumerate(train_iters_t):
                try:
                    target_train_batch = next(t_iter)
                except StopIteration:
                    train_iters_t[idx] = iter(train_loaders_t[idx])
                    target_train_batch = next(train_iters_t[idx])

            target_train_inputs, _, _ = target_train_batch
            target_train_inputs = target_train_inputs.to(device)

            # ----------------------------------------------------------
            # 1. SOURCE SUPERVISED SEGMENTATION STEP
            # ----------------------------------------------------------
            model.train()

            optimizer_seg.zero_grad()

            seg_loss, _ = model(
                x_s=source_train_inputs,
                y_s=source_train_masks,
                mode="seg_source"
            )

            seg_loss.backward()
            optimizer_seg.step()

            # ----------------------------------------------------------
            # 2. DISCRIMINATOR STEP
            # ----------------------------------------------------------
            optimizer_disc.zero_grad()

            disc_loss, _, _ = model(
                x_s=source_train_inputs,
                x_t=target_train_inputs,
                mode="disc"
            )

            disc_loss.backward()
            optimizer_disc.step()

            # ----------------------------------------------------------
            # 3. GENERATOR TARGET ADVERSARIAL STEP
            # ----------------------------------------------------------
            optimizer_gen_t.zero_grad()

            gen_t_loss, _ = model(
                x_t=target_train_inputs,
                mode="gen_t"
            )

            gen_t_loss.backward()
            optimizer_gen_t.step()
            
            # Ensure optimizer grads are cleared safely (use set_to_none=True)
            optimizer_seg.zero_grad(set_to_none=True)
            optimizer_disc.zero_grad(set_to_none=True)
            optimizer_gen_t.zero_grad(set_to_none=True)

            # ----------------------------------------------------------
            # Aggregate train loss for reporting
            # ----------------------------------------------------------
            train_loss = (
                float(seg_loss.detach().cpu()) +
                float(disc_loss.detach().cpu()) +
                float(gen_t_loss.detach().cpu())
            )
             
            train_loss_list.append(float(train_loss))
            train_loss_list_seg.append(float(seg_loss.detach().cpu()))
                
            if 'current_k_fold' in train_parameters:
                nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)
            else:            
                nept_run['training_loss'].log(train_loss)
                nept_run['segmentation_loss'].log(seg_loss.detach().cpu())
            
            for p in model.parameters():
                p.grad = None
                
            # del seg_loss, disc_loss, gen_t_loss, train_loss
            del source_train_inputs, source_train_masks, target_train_inputs
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
                                    
            # -------------------------------
            # Fast Validation (1 batch/domain)
            # -------------------------------    
            if global_step % save_step == 0:        
                model.Gs.eval()
                model.Gt.eval()
                model.D.eval()
                
                with torch.no_grad():                    
                    # --- Source validation (1 batch) ---
                    try:
                        val_batch_s = next(valid_iters_s)
                    except StopIteration:
                        valid_iters_s = iter(valid_loader_s)
                        val_batch_s = next(valid_iters_s)

                    val_imgs_s, val_masks_s, _ = val_batch_s
                    val_imgs_s, val_masks_s = val_imgs_s.to(device), val_masks_s.to(device)

                    # Gs segmenter outputs (no grads)
                    val_preds_s = model.Gs(val_imgs_s)
                    val_loss_s = loss(val_preds_s, val_masks_s).item()
                    val_loss_list_s.append(val_loss_s)
                    
                    # move small example for visualization to CPU numpy (detach)
                    current_img_s = val_imgs_s[0].detach().cpu().numpy()
                    current_mask_s = val_masks_s[0].detach().cpu().numpy()
                    current_pred_s = val_preds_s[0].detach().cpu().numpy()

                    # --- Target validation (1 batch from each domain) ---
                    val_loss_t_list = []
                    val_imgs_t_list = []
                    val_masks_t_list = []
                    val_preds_t_list = []                
                    disc_val_loss_list = []
                    disc_loss_s = 0
                    disc_loss_t = 0
                    
                    # Precompute source features once for Discriminator validation on source features
                    feat_s_val = model.Gs.get_embeddings(val_imgs_s)          # on GPU
                    pred_s_val = model.D(feat_s_val)
                    labels_s_val = torch.ones_like(pred_s_val)  # source label = 1
                    disc_loss_s = model.adv_loss(pred_s_val, labels_s_val).item()
                    
                    # free source disc temporaries asap
                    del pred_s_val, labels_s_val, val_imgs_s, val_masks_s, val_preds_s
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    # iterate target validation iterators (they are iterators over valid loaders)
                    for i, val_iter in enumerate(valid_iters_t):
                        try:
                            val_batch_t = next(val_iter)
                        except StopIteration:
                            valid_iters_t[i] = iter(valid_loaders_t[i])
                            val_batch_t = next(valid_iters_t[i])

                        val_imgs_t, val_masks_t, _ = val_batch_t
                        val_imgs_t, val_masks_t = val_imgs_t.to(device), val_masks_t.to(device)

                        # Target segmenter applied to target images
                        val_preds_t = model.Gt(val_imgs_t)
                        loss_t = loss(val_preds_t, val_masks_t).item()
                        val_loss_t_list.append(loss_t)

                        # Save single example for visualization: detach -> cpu numpy immediately
                        val_imgs_t_list.append(val_imgs_t.detach().cpu().numpy())
                        val_masks_t_list.append(val_masks_t.detach().cpu().numpy())
                        val_preds_t_list.append(val_preds_t.detach().cpu().numpy())
                        
                        # Discriminator validation on target features
                        feat_t_val = model.Gt.get_embeddings(val_imgs_t)   # on GPU
                        pred_t_val = model.D(feat_t_val)
                        labels_t_val = torch.zeros_like(pred_t_val) # target label = 0
                        disc_loss_t += model.adv_loss(pred_t_val, labels_t_val).item()
                        
                        # free per-iteration big tensors ASAP
                        del feat_t_val, pred_t_val, labels_t_val, val_imgs_t, val_masks_t, val_preds_t
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                                                

                    # Average target discriminator loss across target validation domains
                    if len(val_loss_t_list) > 0:
                        val_loss_t = sum(val_loss_t_list) / len(val_loss_t_list)
                        disc_loss_t = disc_loss_t / len(val_loss_t_list)
                    else:
                        val_loss_t = 0.0
                        disc_loss_t = 0.0
                        
                    # Total discriminator validation loss (src + avg tgt)
                    disc_val_loss = disc_loss_s + disc_loss_t

                    val_loss_list_t.append(val_loss_t)
                    disc_val_loss_list.append(disc_val_loss)                   

                    # ----------------------------
                    # Learning Rate Scheduler updates
                    # ----------------------------
                    # 1) segmentation scheduler uses source segmentation val loss
                    lr_plateau_seg.step(val_loss_s)

                    # 2) discriminator scheduler uses adversarial (classification) val loss
                    lr_plateau_disc.step(disc_val_loss)

                    # 3) target-generator scheduler uses a balanced signal (adversarial + target seg)
                    gen_t_val_loss = (disc_loss_t + val_loss_t) / 2.0
                    lr_plateau_gen_t.step(gen_t_val_loss)

                    # Neptune logging (use optimizer_seg lr as canonical)
                    if 'current_k_fold' in train_parameters:
                        nept_run[f'validation_loss_Src_{train_parameters["current_k_fold"]}'].log(val_loss_s)
                        nept_run[f'validation_loss_Tar_{train_parameters["current_k_fold"]}'].log(val_loss_t)
                        nept_run[f'Learning rate_{train_parameters["current_k_fold"]}'].log(optimizer_seg.param_groups[0]['lr'])
                    else:
                        nept_run['validation_loss_Src'].log(val_loss_s)
                        nept_run['validation_loss_Tar'].log(val_loss_t)
                        nept_run['Learning rate'].log(optimizer_seg.param_groups[0]['lr'])

                    # -------------------------------
                    # Saving models + visualizing predictions
                    # -------------------------------
                    if global_step % save_step == 0:

                        # Save ADDA models
                        if "current_k_fold" in train_parameters:
                            torch.save(model.Gs.state_dict(),
                                    model_dir + f'Collagen_Seg_Model_Latest_{train_parameters["current_k_fold"]}.pth')
                            torch.save(model.Gt.state_dict(),
                                    model_dir + f'Collagen_Seg_Model_T_Latest_{train_parameters["current_k_fold"]}.pth')
                        else:
                            torch.save(model.Gs.state_dict(),
                                    model_dir + 'Collagen_Seg_Model_Latest.pth')
                            torch.save(model.Gt.state_dict(),
                                    model_dir + 'Collagen_Seg_Model_T_Latest.pth')

                        # -----------------------
                        # Source domain example
                        # -----------------------                    
                        current_gt_pred_s = np.concatenate(
                            (current_mask_s,
                            current_pred_s),
                            axis=-2
                        )

                        img_dict = {}

                        # -----------------------
                        #  Channel visualizations
                        # -----------------------
                        if isinstance(in_channels, int):

                            if in_channels == 6:
                                current_img_s = np.concatenate(
                                    (current_img_s[:3, :, :], current_img_s[3:, :, :]),
                                    axis=-2
                                )

                                for i in range(len(val_imgs_t_list)):
                                    current_img_t = val_imgs_t_list[i][0]
                                    img_dict[f'Target Images {i + 1}'] = np.concatenate(
                                        (current_img_t[:3, :, :], current_img_t[3:, :, :]),
                                        axis=-2
                                    )
                                    img_dict[f'GT_Pred_Target {i + 1}'] = np.concatenate(
                                        (val_masks_t_list[i][0],
                                        val_preds_t_list[i][0]),
                                        axis=-2
                                    )

                            elif in_channels == 4:
                                current_img_s = np.stack((
                                    np.zeros_like(current_img_s[0]),
                                    current_img_s[0],
                                    np.zeros_like(current_img_s[0]),
                                    current_img_s[1],
                                    current_img_s[2],
                                    current_img_s[3]
                                ), axis=0)
                                current_img_s = np.concatenate((current_img_s[:3], current_img_s[3:]), axis=-2)

                                for i in range(len(val_imgs_t_list)):
                                    current_img_t = val_imgs_t_list[i][0]
                                    current_img_t = np.stack((
                                        np.zeros_like(current_img_t[0]),
                                        current_img_t[0],
                                        np.zeros_like(current_img_t[0]),
                                        current_img_t[1],
                                        current_img_t[2],
                                        current_img_t[3]
                                    ), axis=0)
                                    img_dict[f'Target Images {i + 1}'] = np.concatenate(
                                        (current_img_t[:3], current_img_t[3:]),
                                        axis=-2
                                    )
                                    img_dict[f'GT_Pred_Target {i + 1}'] = np.concatenate(
                                        (val_masks_t_list[i][0],
                                        val_preds_t_list[i][0]),
                                        axis=-2
                                    )

                            elif in_channels == 2:
                                # Create color images for source and target
                                current_img_s = np.stack((
                                    np.zeros_like(current_img_s[0, :, :]),  # R channel DUET image
                                    current_img_s[0, :, :],                 # G channel DUET image
                                    np.zeros_like(current_img_s[0, :, :]),  # B channel DUET image
                                    np.zeros_like(current_img_s[0, :, :]),  # R channel Brightfield image
                                    current_img_s[1, :, :],                 # G channel Brightfield image
                                    np.zeros_like(current_img_s[0, :, :])   # B channel Brightfield image
                                ), axis=0)
                                current_img_s = np.concatenate((current_img_s[:3,:,:], current_img_s[3:,:,:]), axis=-2)
                                for i in range(len(val_imgs_t_list)):
                                    current_img_t = val_imgs_t_list[i][0]
                                    current_img_t = np.stack((
                                        np.zeros_like(current_img_t[0, :, :]),  # R channel DUET image
                                        current_img_t[0, :, :],                 # G channel DUET image
                                        np.zeros_like(current_img_t[0, :, :]),  # B channel DUET image
                                        np.zeros_like(current_img_t[0, :, :]),  # R channel Brightfield image
                                        current_img_t[1, :, :],                 # G channel Brightfield image
                                        np.zeros_like(current_img_t[0, :, :])   # B channel Brightfield image
                                    ), axis=0)                            
                                    
                                    img_dict[f'Target Images {i+1}'] = np.concatenate((current_img_t[:3,:,:],current_img_t[3:,:,:]),axis=-2)
                                    img_dict[f'GT_Pred_Target {i+1}'] = np.concatenate((val_masks_t_list[i][0], val_preds_t_list[i][0]),axis=-2)
                                          
                        # List-style channel encoding
                        elif isinstance(in_channels, list):
                            if sum(in_channels) == 6:
                                current_img_s = np.concatenate((current_img_s[0], current_img_s[1]), axis=-2)
                            elif sum(in_channels) == 2:
                                current_img_s = np.concatenate(
                                    (current_img_s[0][None], current_img_s[1][None]),
                                    axis=-2
                                )

                        # Add source
                        img_dict['Source Images'] = current_img_s
                        img_dict['GT_Pred_Source'] = current_gt_pred_s

                        # Visualization
                        fig = visualize_continuous(img_dict, output_type)

                        # Different saving behavior
                        if output_type == 'comparison':
                            fig.savefig(output_dir + f'/Training_Step_{global_step}_Example.png')
                            nept_run[f'Example_Output_{global_step}'].upload(
                                output_dir + f'/Training_Step_{global_step}_Example.png'
                            )
                        elif output_type == 'prediction':
                            im = Image.fromarray(fig.astype(np.uint8))
                            im.save(output_dir + f'/Training_Step_{global_step}_Example.tif')
                            nept_run[f'Example_Output_{global_step}'].upload(
                                output_dir + f'/Training_Step_{global_step}_Example.tif'
                            )
                            
                    # final cleanup
                    del feat_s_val
                    gc.collect()
                    torch.cuda.empty_cache()
                
            global_step += 1


    if (not global_step % save_step == 0) and (val_loss_t < best_val_loss):
        print("Line 532: not global_step'%'save_step == 0")
        if "current_k_fold" in train_parameters:
            torch.save(model.Gs.state_dict(),model_dir+f'Collagen_Seg_Model_Latest_{train_parameters["current_k_fold"]}.pth')
            torch.save(model.Gt.state_dict(),model_dir+f'Collagen_Seg_Model_T_Latest_{train_parameters["current_k_fold"]}.pth')
        else:
            torch.save(model.Gs.state_dict(),model_dir+'Collagen_Seg_Model_Latest.pth')
            torch.save(model.Gt.state_dict(),model_dir+'Collagen_Seg_Model_T_Latest.pth')

    # Determine the maximum length among the lists
    max_length = max(len(train_loss_list), len(train_loss_list_seg), len(val_loss_list_s), len(val_loss_list_t), len(disc_val_loss_list))
    
    # Pad shorter lists with NaN values to make them equal in length
    train_loss_list_seg += [np.nan] * (max_length - len(train_loss_list_seg))
    disc_val_loss_list += [np.nan] * (max_length - len(disc_val_loss_list))
    train_loss_list += [np.nan] * (max_length - len(train_loss_list))
    val_loss_list_s += [np.nan] * (max_length - len(val_loss_list_s))
    val_loss_list_t += [np.nan] * (max_length - len(val_loss_list_t))

    loss_df = pd.DataFrame(data = {'TrainingLoss':train_loss_list,
                                   'SegmentationLoss': train_loss_list_seg,
                                   'DiscriminatorLoss': disc_val_loss_list,
                                   'ValidationLossSrc': val_loss_list_s,
                                   'ValidationLossTar':val_loss_list_t})
    
    if "current_k_fold" in train_parameters:
        loss_df.to_csv(output_dir+f'_Training_Validation_Loss_{train_parameters["current_k_fold"]}.csv')
        modelsDA = {
            'Source Model': model_dir+f'Collagen_Seg_Model_Latest_{train_parameters["current_k_fold"]}.pth',
            'Target Model': model_dir+f'Collagen_Seg_Model_T_Latest_{train_parameters["current_k_fold"]}.pth'
        }
    else:
        loss_df.to_csv(output_dir+'Training_Validation_Loss.csv')
        modelsDA = {
            'Source Model': model_dir+'Collagen_Seg_Model_Latest.pth',
            'Target Model': model_dir+'Collagen_Seg_Model_T_Latest.pth'
        }
    return modelsDA    