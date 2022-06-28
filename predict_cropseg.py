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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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


def get_individual_fields(segmentation_mask):
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
    solution_writer = solution.reshape(-1)
    for node in range(n_nodes):
        label = labels[node]
        if label < background_label:
            solution_writer[node] = label + 1
        elif label > background_label:
            solution_writer[node] = label

    fields = []
    for i in range(1, np.max(solution)+1):
        x1, y1 = np.min(np.where(solution == i), 1)
        x2, y2 = np.max(np.where(solution == i), 1)
        fields.append(solution[x1:x2, y1:y2])
    return fields


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
    dataset = get_dataset(dataset='crop_seg', filled_mask=True, download=True,
                          split_scheme="cauvery", root_dir=args.path_to_cauvery_images)
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
    output = []
    fields = []
    with torch.no_grad():
        for x, y in test_loader:
            if is_cuda:
                x = x.cuda()
                y = y.cuda()
            for _ in x:
                out = model(_.float())
                out = torch.mean(out, dim=0)
                out = torch.where(out < 0.5, 0, 1)
                output.append(out)
                fields.append(get_individual_fields(out))

            # output = torch.flatten(output)
            # y = torch.flatten(y)
            # idx = torch.where(y == 1)
            # print(idx)
            # output = output[idx]
            predictions.extend(batch_output)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cropseg_weights', type=str,
                        help="Cuda or CPU",
                        default='model_weights/cropseg_weights.pth.tar')
    main(parser.parse_args())