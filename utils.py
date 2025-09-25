import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

def iou_score(output, target):
    smooth = 1e-6
    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1)  # Get the most probable class for each pixel
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_score(output, target):
    smooth = 1e-6
    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1)  # Get the most probable class for each pixel
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()

    return (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)  # Apply softmax to get probabilities for each class
        target = target.squeeze(1)  # Remove the channel dimension from the target tensor
        target = nn.functional.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()  # Convert the target to one-hot encoding

        dice_scores = torch.zeros(self.num_classes).to(output.device)  # Initialize a tensor to store the Dice scores for each class

        for class_index in range(self.num_classes):
            output_class = output[:, class_index]
            target_class = target[:, class_index]
            intersection = torch.sum(output_class * target_class)
            dice_scores[class_index] = (2. * intersection + self.smooth) / (torch.sum(output_class) + torch.sum(target_class) + self.smooth)

        return 1 - torch.mean(dice_scores)  # Return the average Dice loss across all classes


def visualize_prediction(model, dataloader, num_images=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(dataloader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # Select channels 2, 3, 4 and transpose to (H, W, C)
                rgb_image = inputs.cpu().data[j, 2:5, :, :].numpy().transpose((1, 2, 0))
                ax[0].imshow(rgb_image)
                ax[0].title.set_text('Input Image')
                ax[1].imshow(masks.cpu().data[j].numpy(), cmap='gray')
                ax[1].title.set_text('True Mask')
                ax[2].imshow(preds.cpu().data[j].numpy(), cmap='gray')
                ax[2].title.set_text('Predicted Mask')
                plt.show()


# Visualize the encoder output
def visualize_encoder_output(encoder, dataloader, num_images=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(dataloader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = encoder(inputs)
            for j in range(inputs.size()[0]):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # Select channels 2, 3, 4 and transpose to (H, W, C)
                rgb_image = inputs.cpu().data[j, 2:5, :, :].numpy().transpose((1, 2, 0))
                ax[0].imshow(rgb_image)
                ax[0].title.set_text('Input Image')
                ax[1].imshow(masks.cpu().data[j].numpy(), cmap='gray')
                ax[1].title.set_text('True Mask')
                # Visualize the first channel of the encoder output
                ax[2].imshow(outputs.cpu().data[j, 0, :, :], cmap='gray')
                ax[2].title.set_text('Encoder Output')
                plt.show()


# Create a custom dataset and dataloader
class CustomDataset(Dataset):
    def __init__(self, data, masks):
        self.data = data
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index].astype(np.float32)  # Convert data to float32
        mask = self.masks[index].astype(np.int32)  # Convert mask to int32
        return data, mask


def visualize_encoder_output_from_combined_model(model, dataloader, num_images=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(dataloader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model.encoder(inputs)  # Get the encoder output
            for j in range(inputs.size()[0]):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # Select channels 2, 3, 4 and transpose to (H, W, C)
                rgb_image = inputs.cpu().data[j, 2:5, :, :].numpy().transpose((1, 2, 0))
                ax[0].imshow(rgb_image)
                ax[0].title.set_text('Input Image')
                ax[1].imshow(masks.cpu().data[j].numpy(), cmap='gray')
                ax[1].title.set_text('True Mask')
                # Visualize the first channel of the encoder output
                ax[2].imshow(outputs.cpu().data[j, 0, :, :], cmap='gray')
                ax[2].title.set_text('Encoder Output')
                plt.show()