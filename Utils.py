# -*- coding: utf-8 -*-
"""
Utilities included here for collagen segmentation task.  

This includes:

output figure generation, 
metrics calculation,
etc.


"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from PIL import Image

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from sklearn.metrics import roc_auc_score, r2_score
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2lab, lab2rgb

def back_to_reality(tar):
    
    # Getting target array into right format
    classes = np.shape(tar)[-1]
    dummy = np.zeros((np.shape(tar)[0],np.shape(tar)[1]))
    for value in range(classes):
        mask = np.where(tar[:,:,value]!=0)
        dummy[mask] = value

    return dummy

def apply_colormap(img):

    n_classes = np.shape(img)[-1]

    if n_classes==2:
        image = img[:,:,1]
    else:
        image = img[:,:,0]
        for cl in range(1,n_classes):
            image = np.concatenate((image, img[:,:,cl]),axis = 1)

    return image

def visualize_multi_task(images,output_type):
    
    n = len(images)

    if output_type=='comparison':
        fig = plt.figure(constrained_layout = True)
        subfigs = fig.subfigures(1,3)
        image_keys = list(images.keys())
        for outer_ind,subfig in enumerate(subfigs.flat):
            
            current_key = image_keys[outer_ind]

            subfig.suptitle(current_key)

            if len(images[current_key].shape)==4:
                img = images[current_key][0,:,:,:]
            else:
                img = images[current_key]
            
            if np.shape(img)[0]<np.shape(img)[-1]:
                img = np.moveaxis(img,source=0,destination=-1)
            img = np.float32(img)

            if image_keys[outer_ind]=='Image':
                img_ax = subfig.add_subplot(1,1,1)
                img_ax.imshow(img)
            else:
                neg_img = np.uint8(255*np.round(img[:,:,0]))
                coll_img = np.uint8(255*img[:,:,1])


                axs = subfig.subplots(1,2)
                titles = ['Continuous','Binary']
                sub_imgs = [coll_img,neg_img]
                cmaps = ['jet','jet']
                for innerind,ax in enumerate(axs.flat):
                    ax.set_title(current_key+'_'+titles[innerind])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax.imshow(sub_imgs[innerind],cmap=cmaps[innerind])

    elif output_type=='prediction':
        pred_mask = images['Pred_Mask']

        if len(np.shape(pred_mask))==4:
            pred_mask = pred_mask[0,:,:,:]

        pred_mask = np.float32(pred_mask)

        if np.shape(pred_mask)[0]<np.shape(pred_mask)[-1]:
            pred_mask = np.moveaxis(pred_mask,source=0,destination = -1)

        neg_output = 255*np.round(pred_mask[:,:,0])
        coll_output = 255*pred_mask[:,:,1]

        #print(f'Collagen min/max: {np.min(coll_output)},{np.max(coll_output)}')
        #print(f'Negative image min/max: {np.min(neg_output)},{np.max(neg_output)}')

        fig = [coll_output,neg_output]

    return fig


def visualize_continuous(images, output_type):
    """
    Visualizes images neatly in a grid. Applies colormap for prediction masks.
    
    images: dict of image_name -> image_array
    output_type: 'comparison' or 'prediction'
    """
    
    def prepare_image(img):
        img = np.float32(img)
        if len(img.shape) == 4:
            img = img[0]  # take first batch if 4D
        if img.shape[0] < img.shape[-1]:
            img = np.moveaxis(img, 0, -1)
        return img

    if output_type == 'comparison':
        n = len(images)
        # Automatically choose rows and columns
        n_cols = min(n, 4)  # max 3 columns
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = np.array(axes).reshape(-1)  # flatten in case of single row/col

        for i, (key, img_data) in enumerate(images.items()):
            ax = axes[i]
            img = prepare_image(img_data)
            
            if 'Pred' in key or 'GT_Pred' in key or 'Ground_Truth' in key:
                img_to_show = apply_colormap(img)
            else:
                img_to_show = img

            ax.imshow(img_to_show, cmap='jet' if 'Pred' in key else None)
            ax.set_title(key, fontsize=10)
            ax.axis('off')
        
        # Hide any extra axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        return fig

    elif output_type == 'prediction':
        pred_mask = prepare_image(images['Pred_Mask'])
        output_fig = apply_colormap(pred_mask)
        return output_fig

def get_metrics(pred_mask, ground_truth, img_name, calculator, target_type):
    metrics_row = {}

    if target_type == 'binary':
        # ---- Format ground truth ----
        if ground_truth.ndim == 4:
            if ground_truth.shape[1] > 1:
                edited_gt = ground_truth[:, 1, :, :]
            else:
                edited_gt = ground_truth[:, 0, :, :]
            edited_gt = torch.unsqueeze(edited_gt, dim=1)
        else:
            edited_gt = ground_truth[:, :, :]

        # ---- Format prediction ----
        if pred_mask.shape[1] > 1:
            edited_pred = pred_mask[:, 1, :, :]
        else:
            edited_pred = pred_mask[:, 0, :, :]
        edited_pred = torch.unsqueeze(edited_pred, dim=1)

        # ---- Classification metrics (binary masks) ----
        acc, dice, precision, recall, specificity = calculator(
            torch.round(edited_gt), torch.round(edited_pred / 255.0)  # normalize before rounding
        )

        y_true = torch.round(edited_gt).flatten().numpy()
        y_pred = (edited_pred / 255.0).flatten().numpy()  # normalize to [0,1]

        # ---- AUC ----
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_pred)

        # ---- MSE ----
        square_diff = (edited_gt.numpy() - y_pred.reshape(edited_gt.shape)) ** 2
        mse = np.mean(square_diff)

        # ---- R² ----
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan

        metrics_row['Accuracy'] = [round(acc.numpy().tolist(), 4)]
        metrics_row['Dice'] = [round(dice.numpy().tolist(), 4)]
        metrics_row['Precision'] = [round(precision.numpy().tolist(), 4)]
        metrics_row['Recall'] = [round(recall.numpy().tolist(), 4)]
        metrics_row['Specificity'] = [round(specificity.numpy().tolist(), 4)]
        metrics_row['AUC'] = [round(auc, 4)]
        metrics_row['MSE'] = [round(mse, 4)]
        metrics_row['R2'] = [round(r2, 4)]

    elif target_type == 'nonbinary':
        square_diff = (ground_truth.numpy() - pred_mask.numpy()) ** 2
        mse = np.mean(square_diff)
        norm_mse = (square_diff - np.min(square_diff)) / np.max(square_diff)
        norm_mse = np.mean(norm_mse)

        metrics_row['MSE'] = [round(mse, 4)]
        metrics_row['Norm_MSE'] = [round(norm_mse, 4)]

    elif target_type == 'multi_task':
        bin_gt = torch.squeeze(ground_truth[:, 0, :, :])
        bin_pred = pred_mask[0, :, :]

        acc, dice, precision, recall, specificity = calculator(
            bin_gt, torch.round(bin_pred / 255.0)  # normalize before rounding
        )
        metrics_row['Accuracy'] = [round(acc.numpy().tolist(), 4)]
        metrics_row['Dice'] = [round(dice.numpy().tolist(), 4)]
        metrics_row['Precision'] = [round(precision.numpy().tolist(), 4)]
        metrics_row['Recall'] = [round(recall.numpy().tolist(), 4)]
        metrics_row['Specificity'] = [round(specificity.numpy().tolist(), 4)]

        reg_gt = torch.squeeze(ground_truth[:, 1, :, :])
        reg_pred = pred_mask[1, :, :]

        square_diff = (reg_gt.numpy() - reg_pred.numpy()) ** 2
        mse = np.mean(square_diff)
        norm_mse = (square_diff - np.min(square_diff)) / np.max(square_diff)
        norm_mse = np.mean(norm_mse)

        metrics_row['MSE'] = [round(mse, 4)]
        metrics_row['Norm_MSE'] = [round(norm_mse, 4)]

    metrics_row['ImgLabel'] = img_name
    return metrics_row


# Function to resize and apply any condensing transform like grayscale conversion
def resize_special(img,output_size,transform):

    # multi-image input transform
    if 'multi_input' in transform:
        if transform =='multi_input_invbf':
            # Inverting brightfield channels
            img = resize(img,output_shape=(output_size))

            f_img = img[:,:,0:3]
            f_img = f_img/np.sum(f_img,axis=-1)[:,:,None]
            b_img = 255-img[:,:,2:5]
            b_img = b_img/np.sum(b_img,axis=-1)[:,:,None]

            img = np.concatenate((f_img,b_img),axis=-1)
        elif transform =='multi_input_green_invbf':
            # Green channels, inverting bf
            #img = resize(img, output_shape = (output_size))

            f_img = img[:,:,1]
            f_img = (f_img - np.min(f_img))/np.ptp(f_img)
            
            b_img = 255-img[:,:,4]
            b_img = (b_img - np.min(b_img))/np.ptp(b_img)

            img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
            #img = resize(img,output_shape = (output_size))
        elif transform == 'multi_input_mean_invbf':

            # Mean of color channels, inverting bf
            f_img = np.mean(img[:,:,0:3],axis=-1)
            f_img = (f_img - np.min(f_img))/np.ptp(f_img)

            b_img = 255-np.mean(img[:,:,2:5],axis=-1)
            b_img = (b_img - np.min(b_img))/np.ptp(b_img)

            img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
    
        elif transform=='multi_input_green':

            # Grabbing green channels without inverting
            # f_img = img[:,:,1]
            # b_img = img[:,:,4]

            # img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
            img = np.concatenate((img[:,:,1:2], img[:,:,4:5]), axis=-1)
            
        elif transform =='multi_input_green_rgb':
            
            # Grabbing green channel from fluorescent and all channels from brightfield without inverting
            # f_img = img[:,:,1]
            # b_img = img[:,:,3:]

            # img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
            img = np.concatenate((img[:,:,1:2], img[:,:,3:]), axis=-1)
            
        elif transform == 'multi_input_mean':

            # Grabbing mean of brightfield and fluorescent images and concatenating them
            f_img = np.mean(img[:,:,0:3], axis=-1)
            b_img = np.mean(img[:,:,3:6], axis=-1)
            # print(f"shape: {f_img}, {b_img}, {f_img.shape}, {b_img.shape}")
            img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
        elif transform == 'multi_input_rgb':
            # Grabbing all channels from fluorescent and brightfield images
            f_img = img[:,:,0:3]
            b_img = img[:,:,3:6]

            img = np.concatenate((f_img,b_img),axis=-1)

    else:
        if transform=='mean':

            img = np.mean(img,axis = -1)
            img = img[:,:,np.newaxis]

        elif transform in ['red','green','blue']:

            color_list = ['red','green','blue']
            img = img[:,:,color_list.index(transform)]
            img = img[:,:,np.newaxis]

        elif transform == 'rgb2gray':

            img = rgb2gray(img)
            img = img[:,:,np.newaxis]

        elif transform == 'rgb2lab':

            img = rgb2lab(img)
        
        elif type(transform)==dict:

            # Determining non-tissue regions to mask out prior to scaling/conversion
            # For BF images the non-tissue regions are closer to white whereas with fluorescence images they 
            # are closer to black

            lab_img = rgb2lab(img)
            scaled_img = (lab_img-np.nanmean(lab_img))/np.nanstd(lab_img)

            for i in range(3):
                scaled_img[:,:,i] = scaled_img[:,:,i]*transform['norm_std'][i]+transform['norm_mean'][i]
            
            # converting back to rgb
            img = (scaled_img-np.nanmean(scaled_img))/np.nanstd(scaled_img)

        elif transform == 'invert_bf_intensity':
            # Grabbing the green channel from both the fluorescence and brightfield images
            f_green_img = img[:,:,1]
            f_green_img = np.divide(f_green_img,np.sum(img[:,:,0:3],axis=-1),where=(np.sum(img[:,:,0:3],axis=-1)!=0))
            # Inverting brightfield channels
            b_green_inv_img = 255-img[:,:,3]
            b_green_inv_img = np.divide(b_green_inv_img,np.sum(255-img[:,:,2:5],axis=-1),where=(np.sum(255-img[:,:,2:5],axis=-1)!=0))
            img = np.concatenate((f_green_img[:,:,None],b_green_inv_img[:,:,None]),axis=-1)

        elif transform == 'invert_bf_01norm':

            inv_bf = 255-img[:,:,2:5]
            inv_bf_norm = np.divide(inv_bf,np.sum(inv_bf,axis=-1)[:,:,None],where=(np.sum(inv_bf,axis=-1)[:,:,None]!=0))
            f_img = img[:,:,0:3]
            f_norm = np.divide(f_img,np.sum(f_img,axis=-1)[:,:,None],where=(np.sum(f_img,axis=-1)[:,:,None]!=0))

            img = np.concatenate((f_norm,inv_bf_norm),axis=-1)
                
    img = np.float32(resize(img, output_size))

    return img


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad