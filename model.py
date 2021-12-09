# # -*- coding: utf-8 -*-
# """
# Created on Fri Nov 19 17:37:08 2021

# @author: Reihaneh
# """
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric,ConfusionMatrixMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import numpy as np

from data_loader import *
from val_data_loader import * 

device = torch.device("cuda:0")




# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:41:01 2021

@author: Reihaneh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
    
class conv_block_first(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_first,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding='same',bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding='same',bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    
class conv_block_second(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_second,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=2,padding=3,bias=True),
            nn.MaxPool3d(3,stride=1)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv_first(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_first,self).__init__()
        self.up = nn.Sequential(


        nn.ConvTranspose3d(ch_in,ch_out,kernel_size=3,stride=2,padding=1,bias=True,output_padding=1),
        nn.BatchNorm3d(ch_out),
        nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
class up_conv_second(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_second,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
            

        )
        
        

    def forward(self,x):
        x = self.up(x)
        return x
    
class up_conv_last(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_last,self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(48,ch_out,kernel_size=1,stride=1,padding=0),
            nn.Softmax()
		    

        )

    def forward(self,x):
        x = self.up(x)
        return x 
    
    
    
class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        
        # self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout(p=0.2)
        

        self.Conv1 = conv_block_first(ch_in=img_ch,ch_out=48)
        self.Conv2 = conv_block_second(ch_in=48,ch_out=48)
        
        self.Conv3 = conv_block_first(ch_in=48,ch_out=96)
        self.Conv4 = conv_block_second(ch_in=96,ch_out=96)
        
        self.Conv5 = conv_block_first(ch_in=96,ch_out=192)
        self.Conv6 = conv_block_second(ch_in=192,ch_out=192)
        
        self.Conv7 = conv_block_first(ch_in=192,ch_out=384)


        self.Up_8 = up_conv_first(ch_in=384,ch_out=192)
        self.Up_conv9 = up_conv_second(ch_in=384, ch_out=192)
        
        self.Up_10 = up_conv_first(ch_in=192,ch_out=96)
        self.Up_conv11 = up_conv_second(ch_in=192, ch_out=96)
        
        self.Up_12 = up_conv_first(ch_in=96,ch_out=48)
        self.Up_conv13 = up_conv_second(ch_in=96, ch_out=48)
        
        self.Up_14 = up_conv_last(ch_in=48,ch_out=2)
        
        

        


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        #print('Conv1',np.shape(x1))

        x2 = self.Conv2(x1)
        #print('Conv2',np.shape(x2))
        
        x3 = self.Conv3(x2)
        #print('Conv3',np.shape(x3))

        x4 = self.Conv4(x3)
        #print('Conv4',np.shape(x4))
        
        x5 = self.Conv5(x4)
        #print('Conv5',np.shape(x5))
        
        x6 = self.Conv6(x5)
       # x6 = self.dropout(x6)
        #print('Conv6',np.shape(x6))
        
        x7 = self.Conv7(x6)
        #print('Conv7',np.shape(x7))

        # decoding + concat path
        d8 = self.Up_8(x7)
        #print('Convd8',np.shape(d8))
        
        d8 = torch.cat((x5,d8),dim=1)
        #print('catd8,x5',np.shape(d8))
        
        d9 = self.Up_conv9(d8)
        #print('catd9',np.shape(d9))
        
        d10 = self.Up_10(d9)
        #print('catd10',np.shape(d10))
        
        d10 = torch.cat((x3,d10),dim=1)
        #print('catd10,3',np.shape(d10))
        
        d11 = self.Up_conv11(d10)
        #print('catd11',np.shape(d11))

        d12 = self.Up_12(d11)
        #print('catd12',np.shape(d12))
        
        d12 = torch.cat((x1,d12),dim=1)
        #print('catd12',np.shape(d12))

        d13 = self.Up_conv13(d12)
        #print('catd13',np.shape(d13))

        d14 = self.Up_14(d13)
        #print('catd14',np.shape(d14))
        
        

        return d14






# model = U_Net().to(device)
# weights = [0.1, 0.9]
# class_weights = torch.FloatTensor(weights).cuda()
# loss_function = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = torch.optim.Adam(model.parameters(), 1e-4)
# dice_metric = DiceMetric(include_background=False, reduction="mean")


model = U_Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

checkpoint = torch.load(os.path.join(
    root_dir, "best_metric_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss_function = checkpoint['loss']
dice_metric = DiceMetric(include_background=False, reduction="mean")


confusion_matrix=ConfusionMatrixMetric( include_background=False,metric_name=("sensitivity", "precision", "recall",'specificity'))


max_epochs = 16
val_interval = 2
best_metric = -1
best_sensitivity=-1
best_precision=-1
best_recall=-1
best_metric_epoch = -1
best_sensitivity_epoch=-1
best_precision_epoch=-1
best_recall_epoch=-1
epoch_loss_values = []
metric_values = []
sensitivity_values=[]
precision_values=[]
recall_values=[]
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])
y_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])
my_y = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])

# second change
