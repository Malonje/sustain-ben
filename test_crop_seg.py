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
dataset = get_dataset(dataset='crop_seg', filled_mask=False, download=True)

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
def get_fields(segmentation_mask):

    segmentation_mask = segmentation_mask.detach().cpu().numpy().astype(np.uint8)
    row = []
    col = []
    segmentation_mask_reader = segmentation_mask.reshape(-1)
    n_nodes = len(segmentation_mask_reader)
    for node in range(n_nodes):
        idxs = np.unravel_index(node, segmentation_mask.shape)
        if segmentation_mask[idxs] == 0:
            col.append(n_nodes)
        else:
            for i in range(len(idxs)):
                if idxs[i] > 0:
                    new_idxs = list(idxs)
                    new_idxs[i] -= 1
                    new_node = np.ravel_multi_index(new_idxs, segmentation_mask.shape)
                    if segmentation_mask_reader[new_node] != 0:
                        col.append(new_node)
        while len(col) > len(row):
            row.append(node)

    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)
    data = np.ones(len(row), dtype=np.int32)

    graph = csr_matrix((np.array(data), (np.array(row), np.array(col))),
                       shape=(n_nodes + 1, n_nodes + 1))
    n_components, labels = connected_components(csgraph=graph)

    background_label = labels[-1]
    solution = np.zeros(segmentation_mask.shape, dtype=segmentation_mask.dtype)
    # return solution
    # print('SOLUTION:', solution)
    solution_writer = solution.reshape(-1)
    for node in range(n_nodes):
        label = labels[node]
        if label < background_label:
            solution_writer[node] = label + 1
        elif label > background_label:
            solution_writer[node] = label

    return solution

def calc_iou_delineation(pred,true_):
    mask=true_
    predict=pred
    unique_val_mask=np.unique(mask) # find unique value in mask take them except 0
    predict_= get_fields(predict) # label individual fields in prediction
    metrics=[]

    for i in range(0,len(unique_val_mask)):
        mask_val = unique_val_mask[i]
        if mask_val==0:
            continue
        mask_ = np.where(mask==mask_val, 1, 0)      #gives array having value 1 at pos if it is equal to value at that pos otherwise 0
        pred_over_mask = predict_ * mask_          # multiply the mask array for that unique val to prediction arrray
        uniq_pred=np.unique(pred_over_mask)        # find unique vals in the resulting array / unqiue plots to intersect with mask for given val

        for j in range(0,len(uniq_pred)):          # cal iou for each unique val
            uniq_val=uniq_pred[j]
            if uniq_val == 0:
                continue
            pred_array = np.where(predict_==uniq_val, 1, 0)
            #find mlou for mask_ and pred_array
            metrics.append(calc_iou( pred_array , mask_ ))
            # print(pred_array)
            # print(mask_)
    if len(metrics)>0:
        return(sum(metrics)/len(metrics))
    else:
        return 0


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

for x, y in test_loader:
    with torch.no_grad():

        if is_cuda:
            x = x.cuda()
            y = y.cuda()

        y = y[:, 0, :, :]
        fields = []
        output = []
        for k in range(len(x)):
            _ = x[k]
            out = model(_.float())
            out = torch.where(out < 0.5, 0, 1)
            predict_ = get_fields(out)
            # indivual_plots_predicitons=[]
            # for i in range(1, np.max(predict_)+1):
            #     x1, y1 = np.min(np.where(predict_ == i), 1)
            #     x2, y2 = np.max(np.where(predict_ == i), 1)
            #     indivual_plots_predicitons.append(np.where(predict_[x1:x2+1, y1:y2+1] == i, 1, 0))
            # fields.append(indivual_plots_predicitons)
            output.append(calc_iou_delineation(out,y[k]))

        batch_iou.append(sum(output)/len(output))
        # predictions.extend(fields)

print('prediction iou :', sum(batch_iou)/len(batch_iou))













# avg_test_acc = np.round(sum(epoch_test_acc) / len(epoch_test_acc), decimal_precision)
# avg_test_loss = np.round(sum(epoch_test_loss) / len(epoch_test_loss), decimal_precision)
# avg_test_dice_score = np.round(sum(epoch_test_dice) / len(epoch_test_dice), decimal_precision)
# print(f"\t==> Average Test Accuracy: {avg_test_acc}, Average Test Dice. : {avg_test_dice_score}, Average Test loss: {avg_test_loss}")

