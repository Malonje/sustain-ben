import json
import os
import pickle as pkl
import torch
import wandb
import torch
# import utils.logger as logger
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_eval_loader
import torchvision.transforms as transforms
from models.unet import unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
decimal_precision = 5

def get_metric(y_true, y_pred, binarized=True):
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()
    assert(y_true.shape == y_pred.shape)
    if not binarized:
      y_pred[y_pred > 0.5] = 1
      y_pred[y_pred != 1] = 0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    acc = accuracy_score(y_true, y_pred)
    # print('Dice/ F1 score:', f1)
    # print('Accuracy score:', acc)
    # print("Precision recall fscore", precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1))
    return f1, acc

decimal_precision = 5
BACKBONE = 'resnet34'
is_cuda = True
checkpoint_path='/home/parichya/Documents/dilineation_result/'
preprocess_input = get_preprocessing_fn(BACKBONE)
batch_size = 6

num_channels = 3
is_fill = False
is_stacked = False
is_imageNet = True
is_dilated = False
if is_stacked:
    num_channels = 9

dataset = get_dataset(dataset='crop_seg', filled_mask=False)
test_data  = dataset.get_subset('test', transform=transforms.Compose([transforms.Lambda(preprocess_input)]), preprocess_fn=True)
# Prepare the standard data loader
test_loader  = get_eval_loader('standard', test_data,batch_size=batch_size)

best_epoch = 5
input_shape = (224,224,num_channels)


model = None
if is_dilated:
    model = unet_dilated(input_size = input_shape)
elif is_imageNet:
    # model_unet = Unet(BACKBONE, encoder_weights='imagenet')
    model_unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=num_channels, classes=1)

    if is_stacked:
        modules = []
        modules.append(nn.Conv2d(num_channels, 3,  kernel_size=(1,1), padding='same'),nn.ReLU())
        modules.append(model_unet)

        new_model = nn.Sequential(*modules)

        model = new_model

    else:
        model = model_unet
else:
    model = unet(input_size=input_shape)

checkpoint = torch.load(os.path.join(checkpoint_path, f"epoch{best_epoch}.checkpoint.pth.tar"))
model.load_state_dict(checkpoint["model_state_dict"])

if is_cuda:
    model = model.cuda()

model=model.float()
criterion = nn.CrossEntropyLoss()
test_acc=[]
test_dice=[]

for x, y in (test_loader):

    if is_cuda:
        x = x.to("cuda")
        y = y.to("cuda")

        y = y[:,0,:,:]
        output = model(x.float())

        loss = criterion(torch.squeeze(output), y)
        d, acc = get_metric(y.reshape(-1, 1),output.reshape(-1, 1), binarized=False)
        test_acc.append(acc)
        test_dice.append(d)

# test_dice, test_acc = get_metric(y_true.reshape(-1, 1),y_pred.reshape(-1, 1), binarized=False)
# row_data = np.hstack([np.array(y_true).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)])
avg_test_acc = np.round(sum(test_acc) / len(test_acc), decimal_precision)
avg_test_dice = np.round(sum(test_dice) / len(test_dice), decimal_precision)

print(f"* * * Test DICE Score [Using Best Checkpoint]: {avg_test_dice}, Test Accuracy: {avg_test_acc}")
