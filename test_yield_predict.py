from models.gp import GaussianProcess
import json
import os
import pickle as pkl
import torch
import wandb
import torch
# import utils.logger as logger
import torch.nn as nn
from models import convnet
from collections import defaultdict, namedtuple
import segmentation_models_pytorch as smp
import numpy as np
from sustainbench import get_dataset
from pathlib import Path

from sustainbench.common.data_loaders import get_eval_loader
import torchvision.transforms as transforms
# from models.unet import unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from models.loss import l1_l2_loss

decimal_precision = 5




def analyze_results(true, pred, pred_gp):
    """Calculate ME and RMSE"""
    # true = true.numpy().flatten()
    true=np.asarray(true).flatten()
    pred = np.asarray(pred).flatten()
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    me = np.mean(true - pred)

    print(f"Without GP: RMSE: {rmse}, ME: {me}")

    if pred_gp is not None:
        rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
        me_gp = np.mean(true - pred_gp)
        print(f"With GP: RMSE: {rmse_gp}, ME: {me_gp}")
        return rmse, me, rmse_gp, me_gp
    return rmse, me

dropout=0.5
savedir=Path("../")
checkpoint_path = '../model_weights/'

dense_features=None
train_steps=25000
batch_size=32
starter_learning_rate=1e-3
weight_decay=1
l1_weight=1
patience=10
use_gp=False
sigma=1
r_loc=0.5
r_year=1.5
times=32
sigma_e=0.32
sigma_b=0.01
in_channels=3
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_cuda=True
model = convnet.ConvModel(
    in_channels=13,
    dropout=dropout,
    dense_features=dense_features,
    savedir=savedir,
    use_gp=use_gp,
    sigma=sigma,
    r_loc=r_loc,
    r_year=r_year,
    sigma_e=sigma_e,
    sigma_b=sigma_b,
    device=device,
).model
# best_epoch=94
# checkpoint = torch.load(os.path.join(checkpoint_path, f"crop_yield_best_val(fluent-sun-33).pth.tar"))
model.load_state_dict(torch.load('/home/parichya/Documents/model_weights/crop_yield_best_val(fluent-sun-33).pth.tar'))

if is_cuda:
    model = model.cuda()
model=model.float()

dataset = get_dataset(dataset='crop_yield', split_scheme="cauvery", root_dir='data')
train_data = dataset.get_subset('train')
val_data=dataset.get_subset('val')
test_data=dataset.get_subset('test')
batch_size=32
# Prepare the standard data loader
# train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
val_loader   = get_eval_loader('standard', val_data, batch_size=batch_size)
test_loader   = get_eval_loader('standard', test_data, batch_size=batch_size)

running_val_scores = defaultdict(list)
for (val_x, val_y,) in val_loader:
    # print(val_x.min())
    with torch.no_grad():
        model.eval()
        if is_cuda:
            val_x = val_x.to("cuda")
            val_y = val_y.to("cuda")

        val_x = val_x.float()
        val_y = val_y.float()
        val_x = torch.permute(val_x, (0, 3, 1, 2))
        val_pred_y = model(val_x)
        # print(val_pred_y.shape)
        val_pred_y = np.squeeze(val_pred_y)
        val_loss, running_val_scores = l1_l2_loss(
            val_pred_y, val_y, l1_weight, running_val_scores
        )
print("Validation RMSE:", np.array(running_val_scores["RMSE"]).mean())

running_val_scores = defaultdict(list)
for (val_x, val_y,) in test_loader:
    # print(val_x.min())
    with torch.no_grad():
        model.eval()
        if is_cuda:
            val_x = val_x.to("cuda")
            val_y = val_y.to("cuda")

        val_x = val_x.float()
        val_y = val_y.float()
        val_x = torch.permute(val_x, (0, 3, 1, 2))
        val_pred_y = model(val_x)
        # print(val_pred_y.shape)
        val_pred_y = np.squeeze(val_pred_y)
        val_loss, running_val_scores = l1_l2_loss(
            val_pred_y, val_y, l1_weight, running_val_scores
        )
print("Test RMSE:", np.array(running_val_scores["RMSE"]).mean())

exit()

results = defaultdict(list)
# gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)
gp = None
# with torch.no_grad():
    # for train_im, train_yield, train_loc, train_idx, train_year in train_dataloader:
    #     model_output = model(
    #         train_im, return_last_dense=True if (gp is not None) else False
    #     )
    #     if gp is not None:
    #         pred, feat = model_output
    #         if feat.device != "cpu":
    #             feat = feat.cpu()
    #         results["train_feat"].append(feat.numpy())
    #     else:
    #         pred = model_output
    #     results["train_pred"].extend(pred.squeeze(1).tolist())
    #     results["train_real"].extend(train_yield.squeeze(1).tolist())
    #     results["train_loc"].append(train_loc.numpy())
    #     results["train_indices"].append(train_idx.numpy())
    #     results["train_years"].extend(train_year.tolist())

preds=[]
tr=[]
running_val_scores = defaultdict(list)
for test_im, test_yield in val_loader:
    with torch.no_grad():

        if is_cuda:
            test_im=test_im.to("cuda")
            test_yield=test_yield.to("cuda")
        test_im=torch.permute(test_im, (0,3,1,2))
        test_im=test_im.float()
        test_yield=test_yield.float()
        model_output = model(
            test_im#, return_last_dense=True if (gp is not None) else False
        )
        val_pred_y = np.squeeze(model_output)
        val_loss, running_val_scores = l1_l2_loss(
            val_pred_y, test_yield, 1, running_val_scores
        )
        # if gp is not None:
        #     pred, feat = model_output
        #     if feat.device != "cpu":
        #         feat = feat.cpu()
        #     results["test_feat"].append(feat.numpy())
        # else:
        # pred = model_output
        # results["test_pred"].extend(pred.squeeze(1).tolist())
        # results["test_real"].extend(test_yield.tolist())
        # results["test_loc"].append(test_loc.numpy())
        # results["test_indices"].append(test_idx.numpy())
        # results["test_years"].extend(test_year.tolist())

# for key in results:
#     if key in [
#         "train_feat",
#         "test_feat",
#         "train_loc",
#         "test_loc",
#         "train_indices",
#         "test_indices",
#     ]:
#         results[key] = np.concatenate(results[key], axis=0)
#     else:
#         results[key] = np.array(results[key])
        
        
# model_information = {
#             "state_dict": model.state_dict()
#         }
# for key in results:
#     model_information[key] = results[key]
#
# # finally, get the relevant weights for the Gaussian Process
# model_weight = model.state_dict()[model_weight]
# model_bias = model.state_dict()[model_bias]
#
# if model.state_dict()[model_weight].device != "cpu":
#     model_weight, model_bias = model_weight.cpu(), model_bias.cpu()
#
# model_information["model_weight"] = model_weight.numpy()
# model_information["model_bias"] = model_bias.numpy()
#
# if gp is not None:
#     print("Running Gaussian Process!")
#     gp_pred = gp.run(
#         model_information["train_feat"],
#         model_information["test_feat"],
#         model_information["train_loc"],
#         model_information["test_loc"],
#         model_information["train_years"],
#         model_information["test_years"],
#         model_information["train_real"],
#         model_information["model_weight"],
#         model_information["model_bias"],
#     )
#     model_information["test_pred_gp"] = gp_pred.squeeze(1)

# filename = f'{predict_year}_{run_number}_{time}_{"gp" if (gp is not None) else ""}.pth.tar'
# torch.save(model_information, savedir / filename)
print(np.array(running_val_scores["RMSE"]).mean())

results = defaultdict(list)
gp = None
preds = []
tr = []
for test_im, test_yield in test_loader:
    with torch.no_grad():

        if is_cuda:
            test_im = test_im.to("cuda")
            test_yield = test_yield.to("cuda")
        test_im = torch.permute(test_im, (0, 3, 1, 2))
        test_im = test_im.float()
        test_yield = test_yield.float()
        model_output = model(
            test_im  # , return_last_dense=True if (gp is not None) else False
        )
        # if gp is not None:
        #     pred, feat = model_output
        #     if feat.device != "cpu":
        #         feat = feat.cpu()
        #     results["test_feat"].append(feat.numpy())
        # else:
        pred = model_output
        results["test_pred"].extend(pred.squeeze(1).tolist())
        results["test_real"].extend(test_yield.tolist())

print(analyze_results(
    results["test_real"],
    results["test_pred"],
    model_information["test_pred_gp"] if gp is not None else None,
))
