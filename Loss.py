# -*- coding: utf-8 -*-
"""

This is the loss function for Deep-DA-DUET

"""

import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu


def dice_loss(pred, mask, epsilon=1e-6):
    """
    Dice Loss function, calculates the dice coefficient between the predicted and ground truth masks.
    Args:
        pred (torch.Tensor): Predicted tensor (logits or probabilities).
        mask (torch.Tensor): Ground truth mask.
        epsilon (float): Small constant to avoid division by zero.
    Returns:
        torch.Tensor: Dice loss.
    """
    
    intersection = (pred * mask).sum(dim=(2, 3))  # Sum over spatial dimensions (H, W)
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = 1 - (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()  # Return average dice loss over the batch

def structure_loss(pred, mask, beta_1=0.4, beta_2=0.6, epsilon=1e-6):
    # Compute weight map
    # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = torch.ones_like(mask)
    weit = weit / weit.sum(dim=(2, 3), keepdim=True)  # Normalize weights
    
    # Weighted MSE Loss
    wmse = F.mse_loss(pred, mask, reduction='none')
    wmse = (weit * wmse).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    # # IoU Loss with stability    
    # inter = ((pred * mask) * weit).sum(dim=(2, 3))
    # union = ((pred + mask) * weit).sum(dim=(2, 3))
    # wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
    
    # # Final Loss
    # loss = beta_1 * wmse + beta_2 * wiou
    
    # Dice loss
    dice = dice_loss(pred, mask)
    
    # Final Loss
    loss = beta_1 * wmse + beta_2 * dice
    return loss.mean()


# Define a function to compute pairwise distances in batches
def compute_pairwise_distances(x_s, batch_size=1000):
    n = x_s.size(0)
    pairwise_distances = []
    for i in range(0, n, batch_size):
        batch_x_s = x_s[i:i+batch_size]
        distances = torch.pow(batch_x_s.unsqueeze(1) - x_s.unsqueeze(0), 2).sum(2)
        pairwise_distances.append(distances)
    pairwise_distances = torch.cat(pairwise_distances)
    
    # x_s_kernel = torch.exp(-pairwise_distances / (2 * sigma**2)).mean(dim=1)
    
    return pairwise_distances

def mmd_loss(x_s, x_t):
    return torch.abs(x_s.mean() - x_t.mean())

def class_conditional_mmd(pred_s, pred_t):
    # threshold = 0.2
    threshold_s = threshold_otsu(pred_s.detach().cpu().numpy())
    threshold_t = threshold_otsu(pred_t.detach().cpu().numpy())
        
    # Thresholding predictions
    labels_s = torch.where(pred_s >= threshold_s, torch.tensor(1), torch.tensor(0))
    labels_t = torch.where(pred_t >= threshold_t, torch.tensor(1), torch.tensor(0))
    
    # print("label_s.shape:", labels_s.shape)
    
    num_classes = 2  # Assuming binary classification after thresholding
    mmd_losses = []

    for class_idx in range(num_classes):
    
        # Create masks for class_idx
        mask_s = (labels_s == class_idx).float()
        mask_t = (labels_t == class_idx).float()
        # Apply masks to retain the same shape as pred_s and pred_t
        x_s_c = pred_s * mask_s
        x_t_c = pred_t * mask_t
                
        # print("x_s_c.shape:", x_s_c.shape)      
        # # Compute MMD
        # mmd_loss_c = mmd_loss(x_s_c, x_t_c)
        mmd_loss_c = torch.nn.functional.mse_loss(x_s_c, x_t_c)        
        mmd_losses.append(mmd_loss_c)

    # Aggregate all losses
    mean_mmd_loss = torch.stack(mmd_losses).mean()

    return mean_mmd_loss


def adversarial_loss(discriminator, source_features, target_features):
  """
  Calculates domain adversarial loss for domain adaptation.

  Args:
      discriminator: Domain discriminator network (nn.Module).
      segmentation_network: Segmentation network (nn.Module).
      source_features: Feature representations from source domain (tensor).
      target_features: Feature representations from target domain (tensor).

  Returns:
      Domain adversarial loss (tensor).
  """
  # Forward pass through discriminator
  source_preds = discriminator(source_features)
  target_preds = discriminator(target_features)

  # Define loss function (e.g., Binary Cross Entropy)
  criterion = torch.nn.BCELoss()

  # Adversarial loss for source and target domains
  source_loss = criterion(source_preds, torch.ones_like(source_preds))
  target_loss = criterion(target_preds, torch.zeros_like(target_preds))

  # Combine losses
  total_loss = source_loss + target_loss
  return total_loss

# # Example usage:
# # Assuming pred_s, pred_t are torch tensors of grayscale segmentation predictions
# # and labels_s, labels_t are torch tensors of corresponding class labels
# mmd_loss = compute_class_conditional_mmd(pred_s, pred_t, labels_s, labels_t)

# # Incorporate mmd_loss into your total loss
# total_loss = seg_loss + lambda_seg * mmd_loss
