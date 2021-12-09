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

# resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
# md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

# compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
# data_dir = os.path.join(root_dir, "Task09_Spleen")
# if not os.path.exists(data_dir):
#     download_and_extract(resource, compressed_file, root_dir, md5)
    
# train_images = sorted(
#     glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
# train_labels = sorted(
#     glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))







mask_dir ='C:\\reihaneh\\project\\data\\validation\\mask'

mask_names=[]
for x in os.listdir(mask_dir) :
    mask_names.append(os.path.join(mask_dir, x))
    
data_dir='C:\\reihaneh\\project\\data\\validation\\n4'
brain_names=[]
for x in os.listdir(data_dir) :
    brain_names.append(os.path.join(data_dir, x))   
    
train_images= brain_names
train_labels=mask_names

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
val_files = data_dicts[:]


set_determinism(seed=0)



val_transforms = Compose(
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





val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0)#, num_workers=4
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)#, num_workers=4

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=2)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 80], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 80])
plt.show()
