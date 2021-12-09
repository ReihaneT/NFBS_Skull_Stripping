# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:37:37 2021

@author: Reihaneh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:03:20 2021

@author: Reihaneh
"""

from monai.utils import first, set_determinism
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
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from torchsummary import summary
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print_config()
torch.cuda.memory_summary(device=None, abbreviated=False)

directory = os.environ.get("MONAI_DATA_DIRECTORY")
#root_dir = 'C:\\reihaneh\\project\\code\\3d\\3dunet_for_skull_stripping_in_brain_MRI\\monaidice\\with weighted categorical\\second training after tresh1 with tresh 0'
#print(root_dir)

mask_dir ='C:\\reihaneh\\project\\data\\test\\mask'

mask_names=[]
for x in os.listdir(mask_dir) :
    mask_names.append(os.path.join(mask_dir, x))
    
data_dir='C:\\reihaneh\\project\\data\\test\\n4'
brain_names=[]
for x in os.listdir(data_dir) :
    brain_names.append(os.path.join(data_dir, x))   
    
train_images= brain_names
train_labels=mask_names

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
test_files =  data_dicts[:]#data_dicts[:-9], data_dicts[-9:]


set_determinism(seed=0)



test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
               # AsDiscreted(keys='label',to_onehot=2),
            ]
        )





test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=1.0)#, num_workers=4
# val_ds = Dataset(data=val_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1)#, num_workers=4

# first change

