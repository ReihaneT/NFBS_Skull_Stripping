# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:50:56 2021

@author: Reihaneh
"""


import gc
from model import *
from monai.networks import one_hot

#del model

gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_reserved(device=None)
torch.cuda.max_memory_allocated(device=None)
torch.cuda.reset_peak_memory_stats()
max_split_size_mb=1
summary(model,(1,64,64,64))
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = one_hot(labels, num_classes=2, dim=1)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_function},  os.path.join(
                root_dir, "last_metric_model.pth"))
    
    # torch.save(model.state_dict(), os.path.join(
    #     root_dir, "last_metric_model.pth"))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (64, 64, 64)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                confusion_matrix(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            
            sensitivity=confusion_matrix.aggregate()[0].item()
            precision=confusion_matrix.aggregate()[1].item()
            recall=confusion_matrix.aggregate()[2].item()
            # reset the status for next validation round
            dice_metric.reset()
            confusion_matrix.reset()

            metric_values.append(metric)
            sensitivity_values.append(sensitivity)
            precision_values.append(precision)
            recall_values.append(recall)
            
            
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_function},  os.path.join(
                            root_dir, "best_metric_model.pth"))
                
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
            
            
            if sensitivity > best_sensitivity:
                best_sensitivity = sensitivity
                best_sensitivity_epoch = epoch + 1
                
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_function},  os.path.join(
                            root_dir, "best_sensitivity_model.pth"))
                
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current sensitivity: {sensitivity:.4f}"
                f"\nbest sensitivity: {best_sensitivity:.4f} "
                f"at epoch: {best_sensitivity_epoch}"
            )
            
            
            if precision > best_precision:
                best_precision = precision
                best_precision_epoch = epoch + 1
                
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_function},  os.path.join(
                            root_dir, "best_precision_model.pth"))
                
                
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current precision: {precision:.4f}"
                f"\nbest precision: {best_precision:.4f} "
                f"at epoch: {best_precision_epoch}"
            )
            
            
            if recall > best_recall:
                best_recall = recall
                best_recall_epoch = epoch + 1
                
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_function},  os.path.join(
                            root_dir, "best_recall_model.pth"))
                
                
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current recall: {recall:.4f}"
                f"\nbest precision: {best_recall:.4f} "
                f"at epoch: {best_recall_epoch}"
            )
            
print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")



plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()