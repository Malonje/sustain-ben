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
from modelling.unet3d import UNet3D

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
    for module, param in zip(model.center_out.modules(), model.center_out.parameters()):
        param.requires_grad = False
    for module, param in zip(model.dc4.modules(), model.dc4.parameters()):
        param.requires_grad = False
    for module, param in zip(model.trans3.modules(), model.trans3.parameters()):
        param.requires_grad = False
    for module, param in zip(model.dc3.modules(), model.dc3.parameters()):
        param.requires_grad = False
    for module, param in zip(model.final.modules(), model.final.parameters()):
        param.requires_grad = False
    for module, param in zip(model.fn.modules(), model.fn.parameters()):
        param.requires_grad = False
    for module, param in zip(model.logsoftmax.modules(), model.logsoftmax.parameters()):
        param.requires_grad = False
    for module, param in zip(model.logsoftmax.modules(), model.logsoftmax.parameters()):
        param.requires_grad = False

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
            model.load_state_dict(torch.load(pretrained_model_path))

    else:
        raise ValueError(f"Model {model_name} unsupported, check `model_name` arg") 
        

    return model

