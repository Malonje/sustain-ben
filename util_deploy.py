from constants import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import time
import random
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from constants import *
import torch
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="model's name",
                        default='unet3d')


    parser.add_argument('--num_timesteps', type=int, default=184,
                        help="Number of timesteps to include")

    parser.add_argument('--dropout', type=float, default=.5,
                        help="Dropout probability to be used")

    parser.add_argument('--dataset', type=str,
                        help="Full or small?",
                        choices=('full', 'small'),
                        default='full')
    parser.add_argument('--country', type=str,
                        help="country to predict over",
                        default="cauvery")

    parser.add_argument('--include_indices', type=str2bool, default=True,
                        help="Include ndvi and gcvi as input features")

    parser.add_argument('--epochs', type=int, default=130,
                        help="# of times to train over the dataset")
    parser.add_argument('--batch_size', type=int, default=5,
                        help="batch size to use")

    parser.add_argument('--l8_bands', type=str, default=None,
                        help="Number of bands to use from Landsat-8")
    parser.add_argument('--s1_bands', type=str, default=None,
                        help="Number of bands to use from Sentinel-1")
    parser.add_argument('--s2_bands', type=str, default=None,
                        help="Number of bands to use from Sentinel-2")
    parser.add_argument('--ps_bands', type=str, default=None,
                        help="Number of bands to use from PlanetScope")

    parser.add_argument('--optimizer', type=str,
                        help="Optimizer to use for training",
                        default="adam",
                        choices=('sgd', 'adam', 'adam_amsgrad'))
    parser.add_argument('--lr', type=float, default=0.003,
                        help="Initial learning rate to use")
    parser.add_argument('--momentum', type=float,
                        help="Momentum to use when training",
                        default=.9)
    parser.add_argument('--lrdecay', type=float,
                        help="Learning rate decay per **batch**",
                        default=1)

    parser.add_argument('--use_l8', type=str2bool,
                        help="use l8 data?",
                        default=False)
    parser.add_argument('--use_s1', type=str2bool,
                        help="use s1 data?",
                        default=False)
    parser.add_argument('--use_s2', type=str2bool,
                        help="use s2 data?",
                        default=True)
    parser.add_argument('--use_planet', type=str2bool, default=False,
                        help="use planet data?")

    return parser
