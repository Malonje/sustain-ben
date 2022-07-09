"""

File housing all models.

Each model can be created by invoking the appropriate function
given by:

    make_MODELNAME_model(MODEL_SETTINGS)

Changes to allow this are still in progess
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
# import torchfcn
# import fcn

from constants import *
from modelling.unet3d import UNet3D, conv_block, center_in
from util import get_num_bands
import numpy as np

# TODO: figure out how to decompose this

def make_UNet3D_model(n_class, n_channel, timesteps, dropout):
    """ Defined a 3d U-Net model
    Args: 
      n_class - (int) number of classes to predict
      n_channels - (int) number of input channgels

    Returns:
      returns the model!
    """

    model = UNet3D(n_channel, n_class, timesteps, dropout)
    model = model.cuda()
    return model

def make_UNet3D_model_pretrained(n_class, n_channel, timesteps, dropout):
    """ Defined a pretrained (and fixed) 3d U-Net model
        Args:
          n_class - (int) number of classes to predict
          n_channels - (int) number of input channgels

        Returns:
          returns the model!
        """

    model = UNet3D(n_channel, n_class, timesteps, dropout)
    model = model.cuda()
    for module, param in zip(model.en3.modules(), model.en3.parameters()):
        param.requires_grad = False
    for module, param in zip(model.en4.modules(), model.en4.parameters()):
        param.requires_grad = False
    for module, param in zip(model.center_in.modules(), model.center_in.parameters()):
        param.requires_grad = False
    # for module, param in zip(model.center_out.modules(), model.center_out.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.dc4.modules(), model.dc4.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.trans3.modules(), model.trans3.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.dc3.modules(), model.dc3.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.final.modules(), model.final.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.fn.modules(), model.fn.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.logsoftmax.modules(), model.logsoftmax.parameters()):
    #     param.requires_grad = False
    # for module, param in zip(model.logsoftmax.modules(), model.logsoftmax.parameters()):
    #     param.requires_grad = False

    return model


def make_UNetFC_model(n_class, n_channel, timesteps, dropout):
    """ Defined a 3d U-Net model
    Args:
      n_class - (int) number of classes to predict
      n_channels - (int) number of input channgels

    Returns:
      returns the model!
    """
    model = DateExtractor(n_channel, n_class, timesteps, dropout)
    model = model.cuda()
    return model

def get_model(model_name, **kwargs):
    """ Get appropriate model based on model_name and input arguments
    Args: 
      model_name - (str) which model to use 
      kwargs - input arguments corresponding to the model name

    Returns: 
      returns the model!
    """

    model = None

    if model_name == 'unet3d':
        pretrained_model_path = kwargs.get('croptype_weights')
        num_bands = get_num_bands(kwargs)['all']
        print(pretrained_model_path)
        if pretrained_model_path is None:
            model = make_UNet3D_model(n_class=NUM_CLASSES[kwargs.get('country')], n_channel=num_bands, timesteps=kwargs.get('num_timesteps'), dropout=kwargs.get('dropout'))
        else:
            model = make_UNet3D_model_pretrained(n_class=NUM_CLASSES[kwargs.get('country')], n_channel=num_bands, timesteps=kwargs.get('num_timesteps'), dropout=kwargs.get('dropout'))
            pretrained_dict = torch.load(pretrained_model_path)
            state_dict = model.state_dict()
            for k in state_dict:
                if "final" in k:
                    break
                state_dict[k] = pretrained_dict[k]
            model.load_state_dict(state_dict)
    elif model_name == 'unet-fc':
        num_bands = get_num_bands(kwargs)['all']
        model = make_UNetFC_model(n_class=3, n_channel=num_bands,
                                      timesteps=kwargs.get('num_timesteps'), dropout=kwargs.get('dropout'))
    elif model_name == 'unet-fc-yield':
        num_bands = get_num_bands(kwargs)['all']
        model = YieldEstimation(num_bands, 1, timesteps=kwargs.get('num_timesteps'), dropout=kwargs.get('dropout'))
        model = model.cuda()

    else:
        raise ValueError(f"Model {model_name} unsupported, check `model_name` arg") 
        

    return model


class DateExtractor(nn.Module):
    def __init__(self, in_channel, n_classes, timesteps, dropout):
        super(DateExtractor, self).__init__()

        feats = 16
        self.en3 = conv_block(in_channel, feats*4, feats*4)
        self.en4 = conv_block(feats*4, feats*8, feats*8)
        self.center_in = center_in(feats*8, feats*16)
        self.features = nn.Linear(feats*16*7*7, feats*16)
        self.date_predictions = nn.Linear(feats*16, n_classes)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        # print(x.shape)
        en3 = self.en3(x)
        en4 = self.en4(en3)
        center_in = self.center_in(en4)
        # shape
        # print("center in sh", center_in.shape)
        center_in = center_in.permute(0, 2, 1, 3, 4)
        # print("center in sh", center_in.shape)
        shape = center_in.shape
        center_in = center_in.reshape(-1, np.prod(center_in.shape[2:]))
        # shape T X (BXHXW)
        # print("center in sh", center_in.shape)
        center_in = self.dropout(center_in)
        # center_in = center_in.permute(2, 0, 1, 3, 4)
        timestep_features = self.features(center_in)
        # print(timestep_features.shape)
        y = self.date_predictions(timestep_features)
        y = y.reshape(shape[0], shape[1], -1)
        # print(y.shape)
        # print(y)
        y = self.logsoftmax(y)
        # print(y)
        return y


class YieldEstimation(nn.Module):
    def __init__(self, in_channel, n_classes, timesteps, dropout):
        super(YieldEstimation, self).__init__()

        feats = 16
        self.en3 = conv_block(in_channel, feats*4, feats*4)
        self.en4 = conv_block(feats*4, feats*8, feats*8)
        self.center_in = center_in(feats*8, feats*16)
        self.features = nn.Linear(feats*16*7*7*timesteps, n_classes)
        # self.crop_yield = nn.Linear(feats*16*timesteps, n_classes)

        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        # print(x.shape)
        en3 = self.en3(x)
        en4 = self.en4(en3)
        center_in = self.center_in(en4)
        # shape
        # print("center in sh", center_in.shape)
        center_in = center_in.permute(0, 2, 1, 3, 4)
        # print("center in sh", center_in.shape)
        shape = center_in.shape
        center_in = center_in.reshape(-1, np.prod(center_in.shape[1:]))
        # shape T X (BXHXW)
        # print("center in sh", center_in.shape)
        center_in = self.dropout(center_in)
        # center_in = center_in.permute(2, 0, 1, 3, 4)
        timestep_features = self.features(center_in)
        # print(timestep_features.shape)
        # timestep_features = timestep_features.reshape(shape[0], -1)
        # y = self.crop_yield(timestep_features)
        # print(y.shape)
        # print(y)
        # y = self.logsoftmax(y)
        # print(y)
        return timestep_features

