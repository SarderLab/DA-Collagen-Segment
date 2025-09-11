# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:39:49 2024

@author: fafsari

"""

import torch
import torch.nn as nn
import torch.nn.init as init
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation
from torchvision.ops import SqueezeExcitation

        
class EnsembleModel(nn.Module):
    def __init__(self,
                 encoder_name,
                encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate   
        
        if self.active == 'linear':
            self.active = None
            self.final_active = nn.Identity()
        elif self.n_classes > 1:
            self.final_active = nn.Softmax(dim=1)
        elif self.n_classes == 1:
            self.final_active = nn.Sigmoid()
        else:
            self.active = None
            self.final_active = nn.ReLU()
        
        self.model_b = smp.UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )
        
        self.model_d = smp.UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.segmentation_head = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )

    def forward(self,input):

        b_input = input[:, :int(self.in_channels/2), :, :]
        d_input = input[:, int(self.in_channels/2):, :, :]
        # print(f"b_input shape: {b_input.shape}, d_input shape: {d_input.shape}")
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))
        # print(f"b_output shape: {b_output.shape}, d_output shape: {d_output.shape}")
        
        combined_output = torch.cat((b_output,d_output),dim=1)
        # print(f"combined_output shape: {combined_output.shape}")
        
        final_prediction = self.final_active(self.segmentation_head(combined_output))
        # print(f"final_prediction shape: {final_prediction.shape}")
        return final_prediction
    def get_embeddings(self, input):
        b_input = input[:, :int(self.in_channels/2), :, :]
        d_input = input[:, int(self.in_channels/2):, :, :]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))
        combined_output = torch.cat((b_output,d_output),dim=1)
        return combined_output

class SEBlock(nn.Module):
    '''
    SEBlock Class: This class implements the Squeeze-and-Excitation block. 
        1- It reduces the channel dimensions, 
        2- Applies non-linearity, and 
        3- Restores the channel dimensions with a sigmoid activation to get channel-wise attention.
    '''
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class EnsembleModelSE(nn.Module):
    def __init__(self,
                 encoder_name,
                 encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        if self.active == 'sigmoid':
            self.final_active = nn.Sigmoid()
        elif self.active == 'softmax':
            self.final_active = nn.Softmax(dim=1)
        elif self.active == 'linear':
            self.active = None
            self.final_active = nn.Identity()
        else:
            self.active = None
            self.final_active = nn.ReLU()

        self.model_b = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.model_d = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )
        
        # Define SE blocks for each stage
        print("self.model_b.encoder.out_channels:", self.model_b.encoder.out_channels)
        self.se_blocks_b = nn.ModuleList([
            SqueezeExcitation(ch, ch // 8) for ch in self.model_b.encoder.out_channels            
        ])
        self.se_blocks_d = nn.ModuleList([
            SqueezeExcitation(ch, ch // 8) for ch in self.model_d.encoder.out_channels
        ])

        self.segmentation_head = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),#, padding=1),
            nn.Dropout(p=self.dropout_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )

    def forward(self, input):
        b_input = input[:, 0:int(self.in_channels / 2), :, :]        
        d_input = input[:, int(self.in_channels / 2):self.in_channels, :, :]        
                
        # forward inputs to encoders
        b_output = self.model_b.encoder(b_input)
        d_output = self.model_d.encoder(d_input)
        
        # apply SE on all encoder outputs        
        b_output = [se_block(x) for se_block, x in zip(self.se_blocks_b, b_output)]
        d_output = [se_block(x) for se_block, x in zip(self.se_blocks_d, d_output)]
                
        combined_output = torch.cat((b_output, d_output), dim=1)
                        
        final_prediction = self.final_active(self.segmentation_head(combined_output))
        

        return final_prediction
# Ensemble Model with CAM (Contour Attention Map)
class CAM(nn.Module):
    def __init__(self, channels=32):
        super(CAM, self).__init__()
        
        # 3x3 convolution layer
        self.conv3x3_in = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1)
        
        # 1x1 convolution layer
        self.conv1x1_in = nn.Conv2d(channels, 2*channels, kernel_size=1, stride=1, padding=0)
        
        # Global Average Pooling and Global Max Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
                
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
        # Intermediate 1x1 convolution
        self.conv1x1_intermediate = nn.Conv2d(2*channels, 2*channels, kernel_size=1, stride=1, padding=0)
                
        # 3x3 convolution
        self.conv3x3 = nn.Conv2d(2*channels, 2*channels, kernel_size=3, stride=1, padding=1)
        
        # Final 1x1 convolution layer
        self.conv1x1_out = nn.Conv2d(2*channels, 2*channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        
        # First 3x3 convolution
        f_cam_in = self.conv3x3_in(x)
        
        # First 1x1 convolution
        f_ca = self.conv1x1_in(x)
        
        # Global Average and Max Pooling
        avg_pool = self.global_avg_pool(f_ca)
        max_pool = self.global_max_pool(f_ca)        
        
        # Summation of pooled features
        f_ca = avg_pool + max_pool
        
        # Pass through a sigmoid function
        attention_map = self.sigmoid(f_ca)
        
        # Multiply the attention map with the first convolution output
        f_ca = f_cam_in * attention_map
        
        # Pass through the intermediate 1x1 convolution
        f_ca = self.conv1x1_intermediate(f_ca)
        
        # Pass through a 3x3 convolution
        f_cam_out1 = self.conv3x3(f_ca)
        
        # # Pass through the final 1x1 convolution
        # f_cam_out = self.conv1x1_out(f_cam_out1)
        
        # Multiply outputs of 3x3 and 1x1 convolutions
        f_cam_out2 = f_cam_out1 * self.conv1x1_out(f_cam_out1)
        
        # Final output: Add f_mul, f_conv3x3, and f_cam_in
        f_cam_out = f_cam_out1 + f_cam_out2 + f_cam_in
        
        return f_cam_out

class EnsembleModelCAM(nn.Module):
    def __init__(self,
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=2,
                 active='Sigmoid',
                 n_classes=1,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate 
        
        if self.active == 'linear':
            self.active = None
            self.final_active = nn.Identity()
        elif self.n_classes > 1:
            self.final_active = nn.Softmax(dim=1)
        elif self.n_classes == 1:
            self.final_active = nn.Sigmoid()
        else:
            self.active = None
            self.final_active = nn.ReLU()
                
        self.model_b = smp.UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = 3,
            classes = self.n_classes,
            activation = self.active
        )
        
        self.model_d = smp.UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = 1,
            classes = self.n_classes,
            activation = self.active
        )

        self.segmentation_head = nn.Sequential(
            CAM(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )
        
        self.block = nn.Sequential(
            nn.Conv2d(n_classes, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),       
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),       
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1) 
        )            
        
    def forward(self,input):
      
        d_output = self.model_d.decoder(*self.model_d.encoder(input[:, :1, :, :]))
        b_output = self.model_b.decoder(*self.model_b.encoder(input[:, 1:, :, :]))

        # Combined output:
        out = torch.cat((b_output,d_output),dim=1)
        # stage 1 output
        out = self.segmentation_head(out)
        # stage 2 output
        stage2_out = self.block(out)
        # final output the multiplication of stage 1 and stage 2 outputs
        final_prediction = self.final_active(out * stage2_out)
                
        return final_prediction

class EnsembleModelMIT(nn.Module):
    def __init__(self,
                 encoder_name,
                encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate 

        if self.active=='sigmoid':
            self.final_active = nn.Sigmoid()
        elif self.active =='softmax':
            self.final_active = nn.Softmax(dim=1)       
        elif self.active == 'linear':
            self.active = None
            self.final_active = nn.Identity()
        else:
            self.active = None
            self.final_active = nn.ReLU()

        self.model_b = smp.Unet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )
        
        self.model_d = smp.Unet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.segmentation_head = nn.Sequential(
            HeInitLazyConv2d(64, kernel_size=1, bias=False),
            nn.Dropout(p=self.dropout_rate),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )
        # Apply He initialization to convolutional layers
        for layer in self.segmentation_head:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        self.features = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.segmentation_head(self.features))
        
        return final_prediction

class EnsembleModelMAnet(nn.Module):
    def __init__(self,
                 encoder_name,
                encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name    # "mit_b5"
        self.encoder_weights = encoder_weights  # "imagenet"
        self.decoder_channels=(1024, 512, 256, 128, 64)  # Decoder configuration
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate 
        
        if self.active=='sigmoid':
            self.final_active = nn.Sigmoid()
        elif self.active =='softmax':
            self.final_active = nn.Softmax(dim=1)       
        elif self.active == 'linear':
            self.active = None
            self.final_active = nn.Identity()
        else:
            self.active = None
            self.final_active = nn.ReLU()

        self.model_b = smp.MAnet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            decoder_channels = self.decoder_channels,
            decoder_pab_channels = 256,  # Decoder Pyramid Attention Block channels
            in_channels = int(self.in_channels/2),
            classes = self.n_classes
        )
        
        self.model_d = smp.MAnet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            decoder_channels = self.decoder_channels,
            decoder_pab_channels = 256,  # Decoder Pyramid Attention Block channels
            in_channels = int(self.in_channels/2),
            classes = self.n_classes
        )
        
        # Modify all activation functions in the encoder and decoder from ReLU to Mish
        _convert_activations(self.model_b.encoder, nn.ReLU, nn.Mish(inplace=True))
        _convert_activations(self.model_b.decoder, nn.ReLU, nn.Mish(inplace=True))
        
        _convert_activations(self.model_d.encoder, nn.ReLU, nn.Mish(inplace=True))
        _convert_activations(self.model_d.decoder, nn.ReLU, nn.Mish(inplace=True))       
        
        self.segmentation_head = DeepSegmentationHead(
            in_channels=self.decoder_channels[-2], out_channels=1
        )


    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        self.features = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.segmentation_head(self.features))
        
        return final_prediction

class DeepSegmentationHead(nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        # Define a sequence of layers for the segmentation head
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity(),
            Activation(activation) if activation else nn.Identity(),
        ]
        super().__init__(*layers)




class Discriminator(nn.Module):
  """
  Simple Discriminator network for domain adaptation with He initialization.
  """
  def __init__(self, in_channels, hidden_dim):
    super(Discriminator, self).__init__()
    # Define layers for feature extraction and classification
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
    )

    # Apply He initialization to convolutional layers
    for layer in self.layers:
      if isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

  def forward(self, x):
    """
    Forward pass through the Discriminator network.

    Args:
        x: Input feature representation (tensor).

    Returns:
        Discriminator output (tensor).
    """
    return torch.sigmoid(self.layers(x))

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.scale = in_channels ** -0.5

    def forward(self, f_input, b_input):
        # Calculate attention scores
        attn = torch.matmul(f_input, b_input.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn = nn.functional.softmax(attn, dim=-1)
        
        # Attend to values (b_input) using attention weights
        attended_features = torch.matmul(attn, b_input)
        
        return attended_features
    
class AttentionModel(nn.Module):
    def __init__(self, in_channels, active, n_classes):
        super().__init__()

        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes

        encoder = 'mit_b5'
        encoder_weights = 'imagenet'

        if self.active == 'sigmoid':
            self.final_active = nn.Sigmoid()
        elif self.active == 'softmax':
            self.final_active = nn.Softmax(dim=1)
        elif self.active == 'linear':
            self.active = None
            self.final_active = nn.Identity()
        else:
            self.active = None
            self.final_active = nn.ReLU()

        self.model_b = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.model_d = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.attention = CrossAttention(in_channels=64)

        self.segmentation_head = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.Dropout(p=0.1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, kernel_size=1)
        )

    def forward(self, input):
        b_input = input[:, 0:int(self.in_channels / 2), :, :]
        d_input = input[:, int(self.in_channels / 2):self.in_channels, :, :]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        # Apply cross-attention mechanism
        attended_b_output = self.cross_attention(d_output, b_output)

        # Concatenate the attended features
        combined_output = torch.cat((attended_b_output, d_output), dim=1)
        final_prediction = self.final_active(self.segmentation_head(combined_output))

        return final_prediction
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)

class HeInitLazyConv2d(nn.Module):
  """
  Wrapper module for LazyConv2d with He initialization.
  """
  def __init__(self, out_channels, kernel_size=1, bias=False):
    super(HeInitLazyConv2d, self).__init__()
    self.conv = nn.LazyConv2d(out_channels, kernel_size, bias=bias)

  def forward(self, x):
    # He initialization on weight if not already done
    if not hasattr(self.conv, 'weight'):
      init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
    return self.conv(x)

def create_model(model_details, n_classes):
    
    if model_details['architecture']=='Unet++':
        model =  smp.UnetPlusPlus(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            classes = n_classes,
            activation = model_details['active']
            )
        
    elif model_details['architecture']=='ensemble':
        model = EnsembleModel(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],        
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
            )
    elif model_details['architecture']=='ensembleCAM':
        model = EnsembleModelCAM(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],        
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
            )
       
    elif model_details['architecture']=='ensembleMIT':
        model = EnsembleModelMIT(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
            )
    elif model_details['architecture']=='ensembleMAnet':
        model = EnsembleModelMAnet(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
        )
    
    elif model_details['architecture'] == 'ensembleSE':
        model = EnsembleModelSE(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
        )
    
    return model