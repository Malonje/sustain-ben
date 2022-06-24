import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from random import randint
import numpy as np
import pandas as pd
import glob
import math
import warnings
import pdb
import cv2
from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
from sustainbench import logger
import os

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import rasterio
from constants import *
import util
import argparse


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
        albu.Resize(224, 224),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


image_type = 'sentinel'

filepath = 'best-unet-' + image_type
csv_log_file = 'log_unet_' + image_type
batch_size = 6

# Load the full dataset, and download it if necessary

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing_fn(BACKBONE)


def main(args):
    checkpoint_path = args.cropseg_weights
    # Get the training set
    dataset = get_dataset(dataset='crop_seg', filled_mask=False, download=True,
                          split_scheme="cauvery", root_dir=PATH_TO_CAUVERY_IMAGES)
    test_data = dataset.get_subset('test', transform=get_preprocessing(preprocess_input), preprocess_fn=True)

    # Prepare the standard data loader
    test_loader = get_train_loader('standard', test_data, batch_size=batch_size)

    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    model_unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

    modules = [
        model_unet,
        nn.Sigmoid()
    ]
    model = nn.Sequential(*modules)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    is_cuda = True
    model = model.cuda()

    # TESTING
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in test_loader:
            if is_cuda:
                x = x.cuda()
            output = model(x.float())
            output = torch.squeeze(output)
            predictions.append(output)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cropseg_weights', type=str,
                        help="Cuda or CPU",
                        default='model_weights/cropseg_weights.pth.tar')
    main(parser.parse_args())