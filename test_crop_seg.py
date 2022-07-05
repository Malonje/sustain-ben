import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch import Unet
# import keras
# import tensorflow as tf
# from keras.preprocessing import image
# import segmentation_models as sm
# sm.set_framework('tf.keras')
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

# from keras import regularizers
from PIL import Image
from random import randint
# from models.unet import unet
# from models.unet_dilated import unet_dilated
# from utils.data_loader_utils import batch_generator, batch_generator_DG
# from utils.metrics import *
import numpy as np
import pandas as pd
import glob
import math
import warnings
# import keras.backend as K
import pdb
import cv2
from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
from sustainbench import logger
import os

import albumentations as albu
from albumentations.pytorch import ToTensorV2


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


image_type = 'sentinel'

filepath = 'best-unet-' + image_type
csv_log_file = 'log_unet_' + image_type
batch_size = 6

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset='crop_seg', filled_mask=True, download=True)

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing_fn(BACKBONE)
checkpoint_path = 'model_weights/cropseg_weights.pth.tar'

# Get the training set
test_data = dataset.get_subset('test', transform=get_preprocessing(preprocess_input), preprocess_fn=True)

# Prepare the standard data loader
test_loader = get_eval_loader('standard', test_data, batch_size=batch_size)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def learning_rate_scheduler(epoch):
    lr = 1e-4
    '''
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    '''
    print("Set Learning Rate : {}".format(lr))
    return lr


#Set the variables here for training the model
is_fill = False
is_stacked = False
is_imageNet = True
is_dilated = False # dilated models are only for non-pretrained models
image_type = 'sentinel'

num_channels = 3
if is_stacked:
    num_channels = 9

input_shape = (224, 224, num_channels)


if is_dilated:
    # model = unet_dilated(input_size=input_shape)
    pass
elif is_imageNet:
    model_unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

    if is_stacked:
        modules = [
            nn.Conv2d(num_channels, 3, kernel_size=(1, 1), padding='same'),
            nn.ReLU(),
            model_unet
        ]
        model = nn.Sequential(*modules)
    else:
        modules = [
            model_unet,
            nn.Sigmoid()
        ]
        model = nn.Sequential(*modules)
else:
    # model = unet(input_size=input_shape)
    pass

model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

is_cuda = True
if is_cuda:
    model = model.cuda()

decimal_precision = 5
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_scheduler(0))

# TESTING
model.eval()
epoch_test_acc = []
epoch_test_loss = []
epoch_test_dice = []
for x, y in test_loader:
    with torch.no_grad():
        if is_cuda:
            x = x.cuda()
            y = y.cuda()
        output = model(x.float())
        output = torch.squeeze(output)
        y = y[:, 0, :, :]
        loss = criterion(output, y)
        epoch_test_loss.append(loss.item())
        results, resultsstr = dataset.eval(output.detach().cpu().numpy(), y.detach().cpu().numpy(), metadata=None)
        f1, acc, precision_recall = results
        epoch_test_acc.append(acc)
        epoch_test_dice.append(f1)

avg_test_acc = np.round(sum(epoch_test_acc) / len(epoch_test_acc), decimal_precision)
avg_test_loss = np.round(sum(epoch_test_loss) / len(epoch_test_loss), decimal_precision)
avg_test_dice_score = np.round(sum(epoch_test_dice) / len(epoch_test_dice), decimal_precision)
print(f"\t==> Average Test Accuracy: {avg_test_acc}, Average Test Dice. : {avg_test_dice_score}, Average Test loss: {avg_test_loss}")

