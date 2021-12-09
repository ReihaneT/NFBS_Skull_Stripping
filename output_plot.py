# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:40:35 2021

@author: Reihaneh
"""
from model import *
from test_data_loader import *
from torchmetrics import Specificity

# model.load_state_dict(torch.load(
#     os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
sum_sensitivity=0
sum_precision=0
sum_dice=0
sum_specificity=0
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(
            test_data["image"].to(device), roi_size, sw_batch_size, model
        )
        
        test_inputs, test_labels = (
            test_data["image"].to(device),
            test_data["label"].to(device),
        )
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model)
        
        
       
        
        test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
        test_labels = [post_label(i) for i in decollate_batch(test_labels)]
        confusion_matrix(y_pred=test_outputs, y=test_labels)
        sensitivity=confusion_matrix.aggregate()[0].item()
        precision=confusion_matrix.aggregate()[1].item()
        recall=confusion_matrix.aggregate()[2].item()
        specificity=confusion_matrix.aggregate()[3].item()
        dice_metric(y_pred=test_outputs, y=test_labels)
        metric = dice_metric.aggregate().item()
       
        
        sum_sensitivity=sum_sensitivity+sensitivity
        sum_precision=sum_precision+precision
        sum_dice=sum_dice+metric
        sum_specificity=sum_specificity+specificity
        
        print('dice=',metric)
        print('sensitivity=',sensitivity)
        print('precision=',precision)
        print('specificity=',specificity)
        print('**************************')

       # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(test_data["image"][0, 0, :, :, 60], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(test_data["label"][0, 0, :, :, 60])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        # test_outputs = torch.FloatTensor(test_outputs).cuda()
        plt.imshow(torch.argmax(
            test_outputs[0], dim=0).cpu().numpy()[ :, :, 60])
        plt.show()
        
        
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(test_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(test_data["label"][0, 0, :, :, 80])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        # test_outputs = torch.FloatTensor(test_outputs).cuda()
        plt.imshow(torch.argmax(
            test_outputs[0], dim=0).cpu().numpy()[ :, :, 80])
        plt.show()
        
        
        
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(test_data["image"][0, 0, :, :, 70], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(test_data["label"][0, 0, :, :, 70])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        # test_outputs = torch.FloatTensor(test_outputs).cuda()
        plt.imshow(torch.argmax(
            test_outputs[0], dim=0).cpu().numpy()[ :, :, 70])
        plt.show()
        
        
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(test_data["image"][0, 0, :, :, 50], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(test_data["label"][0, 0, :, :, 50])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        # test_outputs = torch.FloatTensor(test_outputs).cuda()
        plt.imshow(torch.argmax(
            test_outputs[0], dim=0).cpu().numpy()[ :, :, 50])
        plt.show()
        
        # if i == 5:
        #     break
print('all dice=',sum_dice/15)
print('all sensitivity=',sum_sensitivity/15)
print('all precision=',sum_precision/15)
print('all specificity=',sum_specificity/15)
