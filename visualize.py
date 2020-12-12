from __future__ import print_function
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sn

from deep_emotion import Deep_Emotion
from data_loaders import Plain_Dataset, eval_data_dataloader

parser = argparse.ArgumentParser(description="Configuration of testing process")
parser.add_argument('-d', '--data', type=str,required = True, help='Path to the finaltest folder that contains finaltest images')
parser.add_argument('-c', '--file', type=str,required = True, help='Path to the finaltest.csv')
parser.add_argument('-f', '--model', type=str,required = True, help='Path to pretrained model')
parser.add_argument('-t', '--test_acc', action='store_true', help='Returns test accuracy and visualization of confusion matrix')
parser.add_argument('-s', '--saliency_map', action='store_true', help='Returns saliency map for 10 test images')
parser.add_argument('-m', '--channel50', type=bool, help= 'number of channel')
parser.add_argument('-n', '--stn', type=bool, help= 'if you do not want to use stn, type False')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transformation = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5))
])

dataset = Plain_Dataset(
    csv_file=args.file, 
    img_dir=args.data, 
    datatype='finaltest',
    transform=transformation
)

if args.channel50:
    num_channel = 50
else:
    num_channel = 10

if args.stn:
    stn = args.stn
else:
    stn = True

net = Deep_Emotion(num_channel, stn)
net.load_state_dict(torch.load(args.model))
net.to(device)
net.eval()

classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')

if args.test_acc:
    test_loader = DataLoader(dataset, batch_size=64, num_workers=0)
    total = []
    confusion_matrix = torch.zeros((len(classes), len(classes)), device=device)

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)

            # Test Accuracy Calculation
            wrong = torch.where(classs != labels, torch.tensor([1.], device=device), torch.tensor([0.], device=device))
            acc = 1 - (torch.sum(wrong) / data.size(0))
            total.append(acc.item())

            # Confusion Matrix Calculation
            one_hot_class = F.one_hot(classs, num_classes=len(classes)).float()
            one_hot_labels = F.one_hot(labels, num_classes=len(classes)).float()
            # conf = (num_classes, B) x (B, num_classes)
            conf = torch.matmul(torch.transpose(one_hot_class, 0, 1), one_hot_labels)
            confusion_matrix += conf

    confusion_matrix = confusion_matrix.cpu().numpy()
    confusion_matrix = pd.DataFrame(confusion_matrix, classes, classes)

    print('Test Accuracy: {}'.format(100 * np.mean(total)))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.2)  # for label size
    sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 14}, cmap='YlGnBu')  # font size, colormap
    plt.show()

if args.saliency_map:
    img_dim = 256
    mask_dim = 56  # Must be a multiple of 8
    stride = 8
    mask_channels = (img_dim - mask_dim) // stride + 1  # A total of mask_channels^2 masks
    print('Creating saliency map with mask_dim: {} and stride: {}'.format(mask_dim, stride))
    print('Number of masked images to be evaluated: {}'.format(mask_channels**2))

    saliency_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    def create_mask(img_dim, mask_dim, stride):
        masks = torch.zeros((mask_channels**2, img_dim, img_dim), device=device)

        for C in range(mask_channels**2):
            i = C % mask_channels
            j = C // mask_channels
            mask = masks[C,:,:]
            mask[i*stride:i*stride+mask_dim-1,j*stride:j*stride+mask_dim-1] = 2
            masks[C,:,:] = mask

        return masks

    def create_saliency_map(img_dim, mask_dim, stride, wrong_labels, saliency_max=10):
        mask_channels = (img_dim - mask_dim) // stride + 1  # total of mask_channels^2 masks
        saliency_map = torch.zeros((img_dim, img_dim), device=device)

        for C in range(mask_channels**2):
            i = C % mask_channels
            j = C // mask_channels
            wrong_label = wrong_labels[C]
            if wrong_label:  # add one to saliency map
                saliency_map[i*stride:i*stride+mask_dim-1,j*stride:j*stride+mask_dim-1] += 1
            
        # Normalize to 0 ~ 1
        saliency_map = saliency_map / (mask_dim // stride)**2

        return saliency_map



    max_images = 1

    num_images = 0
    masks = create_mask(img_dim, mask_dim, stride)
    with torch.no_grad():
        for data, label in saliency_loader:
            data, labels = data.to(device), label.to(device)
            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)

            # If test image is not predicted correctly, no point in making a saliency map
            if classs[0] != label[0]:
                continue

            masked_data = torch.clamp(data - masks.unsqueeze(1), min=-1)
            masked_outputs = net(masked_data)
            masked_pred = F.softmax(masked_outputs, dim=1)
            masked_class = torch.argmax(masked_pred, 1)

            wrong_labels = masked_class != labels

            if torch.sum(wrong_labels) > 0.3*mask_channels**2:
                continue

            saliency_map = create_saliency_map(img_dim, mask_dim, stride, wrong_labels)
            
            # Change the range of image pixel values back to 0~1
            unnormalized_data = (data + 1) / 2  

            weighted_data = torch.clamp(unnormalized_data + 0.8*saliency_map, max=1)
            saliency_map = saliency_map.squeeze().cpu()
            weighted_data = weighted_data.squeeze().cpu()

            plt.style.use('classic')
            plt.rcParams['figure.figsize'] = (10,4)
            
            # Show saliency_map and the saliency-weighted image
            plt.imshow(saliency_map.numpy(), cmap=plt.cm.gray)
            plt.show()

            # Show original image
            plt.imshow(data.squeeze().cpu().numpy(), cmap=plt.cm.gray)
            plt.show()

            # Show saliency_map + orignal_image
            plt.imshow(weighted_data.numpy(), cmap=plt.cm.gray)            
            plt.show()

            num_images += 1

            if max_images <= num_images:
                break
