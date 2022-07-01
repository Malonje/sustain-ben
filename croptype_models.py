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
    if model_name == 'unet-fc':
        num_bands = get_num_bands(kwargs)['all']
        model = make_UNetFC_model(n_class=3, n_channel=num_bands,
                                      timesteps=kwargs.get('num_timesteps'), dropout=kwargs.get('dropout'))


    else:
        raise ValueError(f"Model {model_name} unsupported, check `model_name` arg") 
        

    return model


class DateExtractor(nn.Module):
    def __init__(self, in_channel, n_classes, timesteps, dropout):
        super(DateExtractor, self).__init__()

        feats = 16
        self.en3 = conv_block(in_channel, feats*4, feats*4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats*4, feats*8, feats*8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats*8, feats*16)

        self.features = nn.Linear(timesteps, feats*8)
        self.date_predictions = nn.Linear(timesteps, n_classes)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        en3 = self.en3(x)
        pool_3 = self.pool_3(en3)
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        center_in = self.center_in(pool_4)
        center_in = center_in.permute(0, 2, 1, 3, 4)
        shape_num = final.shape[0:2]
        center_in = center_in.reshape(-1, np.prod(final.shape[2:]))
        center_in = self.dropout(center_in)
        timestep_features = self.features(center_in)
        y = self.date_predictions(timestep_features)
        y = self.logsoftmax(y)
        return y

