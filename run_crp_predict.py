from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
from models import convnet
# from sustainbench.models.feassture_engineering import Engineer
from models.loss import l1_l2_loss
from sustainbench import logger
import os
from collections import defaultdict, namedtuple
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score




checkpoint_path='/home/parichya/Documents/predict_result/'

run_name = logger.init(project='crop_prediction', reinit=True)

# Get the training set
# cleaned_data_path="data/img_output",
# yield_data_path="data/yield_data.csv",
# county_data_path="data/county_data.csv",
# num_bins=32,
# max_bin_val=4999,
# cleaned_data_path = Path(cleaned_data_path)
# yield_data_path = Path(yield_data_path)
# county_data_path = Path(county_data_path)
#
# engineer = Engineer(cleaned_data_path, yield_data_path, county_data_path)
# engineer.process(
#     num_bands=9,
#     generate="histogram",
#     num_bins=num_bins,
#     max_bin_val=max_bin_val,
#     channels_first=True,
# )


dataset = get_dataset(dataset='crop_yield')
train_data = dataset.get_subset('train')#, transform=transforms.Compose([transforms.Lambda(preprocess_input)]), preprocess_fn=True)
val_data   = dataset.get_subset('val')#, transform=transforms.Compose([transforms.Lambda(preprocess_input)]), preprocess_fn=True)
batch_size=32
# Prepare the standard data loader
train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
val_loader   = get_train_loader('standard', val_data, batch_size=batch_size)
dropout=0.5
savedir=Path("/home/parichya/Documents/sustainbench/results_prediction")
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
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = convnet.ConvModel(
    in_channels=9,
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
)
# if device.type != "cpu":
is_cuda=True
if is_cuda:
    model = model.model.cuda()

# model_type = model_type
# model_weight = model_weight
# model_bias = model_bias



optimizer = torch.optim.Adam(
            [pam for pam in model.parameters()],
            lr=starter_learning_rate,
            weight_decay=weight_decay,
        )

#EPOCHS
# num_epochs = int(train_steps / (32 / batch_size))#int(train_steps / (train_images.shape[0] / batch_size))
num_epochs=100
print(f"Training for {num_epochs} epochs")

train_scores = defaultdict(list)
val_scores = defaultdict(list)

step_number = 0
min_loss = np.inf
best_state = model.state_dict()
model=model.float()



if patience is not None:
    epochs_without_improvement = 0

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/'{num_epochs}']" )
    model.train()

    # running train and val scores are only for printing out
    # information
    running_train_scores = defaultdict(list)

    for train_x, train_y in train_loader:
        optimizer.zero_grad()
        print('train_X size:',train_x.shape)
        if is_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        train_x=torch.permute(train_x, (0,3,1,2))
        # print(train_y.min())

        train_x=train_x.float()
        train_y=train_y.float()
        pred_y = model(train_x)
        pred_y=np.squeeze(pred_y)
        # print('pred:',(pred_y).dtype)
        # print('y:',(train_y).dtype)
        loss, running_train_scores = l1_l2_loss(
            pred_y, train_y, l1_weight, running_train_scores
        )

        loss.backward()
        optimizer.step()

        train_scores["loss"].append(loss.item())

        step_number += 1

        if step_number in [4000, 20000]:
            for param_group in optimizer.param_groups:
                param_group["lr"] /= 10

    train_output_strings = []
    for key, val in running_train_scores.items():
        train_output_strings.append(
            "{}: {}".format(key, round(np.array(val).mean(), 5))
        )

    running_val_scores = defaultdict(list)
    checkpoint = {
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                }
    torch.save(checkpoint, os.path.join(checkpoint_path, f"epoch{epoch+1}.checkpoint.pth.tar"))
    model.eval()


    for (val_x,val_y,) in val_loader:
        # print(val_x.min())
        with torch.no_grad():
            if is_cuda:
                val_x=val_x.to("cuda")
                val_y=val_y.to("cuda")


            val_x=val_x.float()
            val_y=val_y.float()
            val_x=torch.permute(val_x, (0,3,1,2))
            val_pred_y = model(val_x)
            # print(val_pred_y.shape)
            val_pred_y=np.squeeze(val_pred_y)
            val_loss, running_val_scores = l1_l2_loss(
                val_pred_y, val_y, l1_weight, running_val_scores
            )

            val_scores["loss"].append(val_loss.item())

    val_output_strings = []
    for key, val in running_val_scores.items():
        val_output_strings.append(
            "{}: {}".format(key, round(np.array(val).mean(), 5))
        )
    # print(float(train_output_strings[0].split(':')[1])
    print("TRAINING: {}".format(", ".join(train_output_strings)))
    print("VALIDATION: {}".format(", ".join(val_output_strings)))

    logger.log({
        f"Train RMSE": float(train_output_strings[2].split(':')[1]),
        f"Train L2": float(train_output_strings[0].split(':')[1]),
        f"Train loss" : float(train_output_strings[1].split(':')[1]),
        f"Val RMSE": float(train_output_strings[2].split(':')[1]),
        f"Val L2" : float(val_output_strings[0].split(':')[1]),
        f"Val Loss" : float(val_output_strings[1].split(':')[1]),

    })


    epoch_val_loss = np.array(running_val_scores["loss"]).mean()

    if epoch_val_loss < min_loss:
        best_state = model.state_dict()
        min_loss = epoch_val_loss

        if patience is not None:
            epochs_without_improvement = 0
    elif patience is not None:
        epochs_without_improvement += 1

        # if epochs_without_improvement == patience:
        #     # revert to the best state dict
        #     model.load_state_dict(best_state)
        #     print("Early stopping!")
        #     break

model.load_state_dict(best_state)
print( train_scores, val_scores)




















# model.eval()
#         with torch.no_grad():
#             for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
#                 train_dataloader
#             ):
#                 model_output = model(
#                     train_im, return_last_dense=True if (gp is not None) else False
#                 )
#                 if gp is not None:
#                     pred, feat = model_output
#                     if feat.device != "cpu":
#                         feat = feat.cpu()
#                     results["train_feat"].append(feat.numpy())
#                 else:
#                     pred = model_output
#                 results["train_pred"].extend(pred.squeeze(1).tolist())
#                 results["train_real"].extend(train_yield.squeeze(1).tolist())
#                 results["train_loc"].append(train_loc.numpy())
#                 results["train_indices"].append(train_idx.numpy())
#                 results["train_years"].extend(train_year.tolist())
#
#





        #     for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
        #         test_dataloader
        #     ):
        #         model_output = model(
        #             test_im, return_last_dense=True if (gp is not None) else False
        #         )
        #         if gp is not None:
        #             pred, feat = model_output
        #             if feat.device != "cpu":
        #                 feat = feat.cpu()
        #             results["test_feat"].append(feat.numpy())
        #         else:
        #             pred = model_output
        #         results["test_pred"].extend(pred.squeeze(1).tolist())
        #         results["test_real"].extend(test_yield.squeeze(1).tolist())
        #         results["test_loc"].append(test_loc.numpy())
        #         results["test_indices"].append(test_idx.numpy())
        #         results["test_years"].extend(test_year.tolist())
        #
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
        # return results
