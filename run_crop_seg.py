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
checkpoint_path = 'model_weights/'

run_name = logger.init(project='dilineation', reinit=True)

# Get the training set
train_data = dataset.get_subset('train', transform=get_preprocessing(preprocess_input), preprocess_fn=True)
val_data = dataset.get_subset('val', transform=get_preprocessing(preprocess_input), preprocess_fn=True)

# Prepare the standard data loader
train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
val_loader = get_train_loader('standard', val_data, batch_size=batch_size)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


# def get_metric(y_true, y_pred, binarized=True):
#     y_true = y_true.detach().cpu().numpy().flatten()
#     y_pred = y_pred.detach().cpu().numpy().flatten()
#     assert(y_true.shape == y_pred.shape)
#     if not binarized:
#       y_pred[y_pred > 0.5] = 1
#       y_pred[y_pred != 1] = 0
#     y_true = y_true.astype(int)
#     y_pred = y_pred.astype(int)
#     f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
#     acc = accuracy_score(y_true, y_pred)
#     # print('Dice/ F1 score:', f1)
#     # print('Accuracy score:', acc)
#     # print("Precision recall fscore", precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1))
#     return f1, acc


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

is_cuda = True
if is_cuda:
    model = model.cuda()

decimal_precision = 5
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_scheduler(0))

# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(lr=learning_rate_scheduler(0)),
#               metrics=['acc', f1])

# checkpoint = ModelCheckpoint(filepath, monitor='f1', verbose=1, save_best_only=True, mode='max')
# csv_logger = CSVLogger(csv_log_file, append=True, separator=';')
# callbacks_list = [checkpoint, csv_logger]
# train_df = pd.read_csv('/home/parichya/Documents/sustainbench/data/crop_delineation/train_df.csv')
# val_df = pd.read_csv('/home/parichya/Documents/sustainbench/data/crop_delineation/val_df.csv')
# model = model.float()

# Train loop
epochs = 200
for i in range(epochs):
    model.train()
    epoch_train_acc = []
    epoch_train_loss = []
    epoch_dice = []
    for x, y_true in train_loader:
        if is_cuda:
            x = x.cuda()
            y_true = y_true.cuda()

        output = model(x.float())
        output = torch.squeeze(output)

        y_true = y_true[:, 0, :, :]

        loss = criterion(output, y_true)
        epoch_train_loss.append(loss.item())

        results, resultsstr = dataset.eval(output.detach().cpu().numpy(), y_true.detach().cpu().numpy(), metadata=None)
        f1, acc, precision_recall = results
        print(resultsstr)
        epoch_train_acc.append(acc)
        epoch_dice.append(f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    checkpoint = {
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                }
    torch.save(checkpoint, os.path.join(checkpoint_path, f"epoch{i}.checkpoint.pth.tar"))

    avg_train_acc = np.round(sum(epoch_train_acc) / len(epoch_train_acc), decimal_precision)
    avg_train_loss = np.round(sum(epoch_train_loss) / len(epoch_train_loss), decimal_precision)
    avg_train_dice_score = np.round(sum(epoch_dice) / len(epoch_dice), decimal_precision)
    print(f"Epoch [{i + 1}/'{epochs}'] Average Train Accuracy: {avg_train_acc},Avergae Dice Score: {avg_train_dice_score}, Average Train Loss: {avg_train_loss}")

    #VALIDATION
    model.eval()
    epoch_val_acc = []
    epoch_val_loss = []
    epoch_val_dice = []
    for x, y in val_loader:
        with torch.no_grad():
            if is_cuda:
                x = x.cuda()
                y = y.cuda()
            output = model(x.float())
            output = torch.squeeze(output)
            y = y[:, 0, :, :]
            loss = criterion(output, y)
            epoch_val_loss.append(loss.item())
            results, resultsstr = dataset.eval(output.detach().cpu().numpy(), y.detach().cpu().numpy(), metadata=None)
            f1, acc, precision_recall = results
            epoch_val_acc.append(acc)
            epoch_val_dice.append(f1)


    # val_accuracy, val_precision, val_recall, val_f1, val_iou = get_metric(y_true, y_pred)
    avg_val_acc = np.round(sum(epoch_val_acc) / len(epoch_val_acc), decimal_precision)
    avg_val_loss = np.round(sum(epoch_val_loss) / len(epoch_val_loss), decimal_precision)
    avg_val_dice_score = np.round(sum(epoch_val_dice) / len(epoch_val_dice), decimal_precision)
    print(f"\t==> Average Val. Accuracy: {avg_val_acc}, Average Val. Dice. : {avg_val_dice_score}, Average Val. loss: {avg_val_loss}")

    logger.log({
        f"Average Train Accuracy": avg_train_acc,
        f"Avergae Dice Score" : avg_train_dice_score,
        f"Average Train Loss" : avg_train_loss,
        f"Average Val Train Accuracy" : avg_val_acc,
        f"Average Val Dice Score" : avg_val_dice_score,
        f"Average Val Loss" : avg_val_loss,

    })



# # Get predictions for the full test set
# all_y_true=[]
# all_y_pred=[]
# for x_test, y_test in test_loader:
#     y_pred = model(x_test)
#     all_y_pred.append(y_pred)
#     all_y_true.append(y_test)
#
# # Evaluate
# print(dataset.eval(all_y_pred, all_y_true))
# # {'recall_macro_all': 0.66, ...}




