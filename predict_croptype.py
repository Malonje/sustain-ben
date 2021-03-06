"""

Script for training and evaluating a model

"""
from cmath import nan
import os
import loss_fns
import croptype_models
import datetime
import torch
# import datasets
import metrics
import util
import numpy as np
import pickle

from torch import autograd

from constants import *
from tqdm import tqdm
from torch import autograd
import visualize

from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import torch.nn.functional as F


def evaluate_split(model, model_name, split_loader, device, loss_weight, weight_scale, gamma, num_classes, country,
                   var_length):
    total_loss = 0
    total_pixels = 0
    total_cm = np.zeros((num_classes, num_classes)).astype(int)
    loss_fn = loss_fns.get_loss_fn(model_name)
    for inputs, targets, cloudmasks, hres_inputs in split_loader:
        with torch.set_grad_enabled(False):
            if not var_length:
                inputs.to(device)
            else:
                for sat in inputs:
                    if "length" not in sat:
                        inputs[sat].to(device)
            targets.to(device)
            hres_inputs.to(device)
            if hres_inputs is not None: hres_inputs.to(device)

            preds = model(inputs, hres_inputs) if model_name in MULTI_RES_MODELS else model(inputs)
            batch_loss, batch_cm, _, num_pixels, confidence = evaluate(model_name, preds, targets, country,
                                                                       loss_fn=loss_fn, reduction="sum",
                                                                       loss_weight=loss_weight,
                                                                       weight_scale=weight_scale, gamma=gamma)
            total_loss += batch_loss.item()
            total_pixels += num_pixels
            total_cm += batch_cm

    f1_avg = metrics.get_f1score(total_cm, avg=True)
    acc_avg = sum([total_cm[i][i] for i in range(num_classes)]) / np.sum(total_cm)
    return total_loss / total_pixels, f1_avg, acc_avg


def evaluate(model_name, preds, labels, country, loss_fn=None, reduction=None, loss_weight=None, weight_scale=None,
             gamma=None):
    """ Evalautes loss and metrics for predictions vs labels.

    Args:
        preds - (tensor) model predictions
        labels - (npy array / tensor) ground truth labels
        loss_fn - (function) function that takes preds and labels and outputs some loss metric
        reduction - (str) "avg" or "sum", where "avg" calculates the average accuracy for each batch
                                          where "sum" tracks total correct and total pixels separately
        loss_weight - (bool) whether we use weighted loss function or not

    Returns:
        loss - (float) the loss the model incurs
        cm - (nparray) confusion matrix given preds and labels
        accuracy - (float) given "avg" reduction, returns accuracy
        total_correct - (int) given "sum" reduction, gives total correct pixels
        num_pixels - (int) given "sum" reduction, gives total number of valid pixels
    """
    cm = metrics.get_cm(preds, labels, country, model_name)

    if model_name in NON_DL_MODELS:
        accuracy = metrics.get_accuracy(model_name, preds, labels, reduction=reduction)
        return None, cm, accuracy, None
    elif model_name in DL_MODELS:
        if reduction == "avg":
            loss, confidence = loss_fn(labels, preds, reduction, country, loss_weight, weight_scale)

            accuracy = metrics.get_accuracy(model_name, labels, model_name, reduction=reduction)
            return loss, cm, accuracy, confidence
        elif reduction == "sum":
            loss, confidence, _ = loss_fn(labels, preds, reduction, country, loss_weight, weight_scale)
            total_correct, num_pixels = metrics.get_accuracy(model_name, preds, labels, reduction=reduction)
            return loss, cm, total_correct, num_pixels, confidence
        else:
            raise ValueError(f"reduction: `{reduction}` not supported")


def predict_dl_model(model, model_name, dataloaders, args):
    splits = ['train', 'val'] if not args.eval_on_test else ['test']
    predictions = []

    if args.clip_val:
        clip_val = sum(p.numel() for p in model.parameters() if p.requires_grad) // 20000
        print('clip value: ', clip_val)

    # set up information lists for visdom
    vis_logger = visualize.VisdomLogger(args.env_name, model_name, args.country, splits)
    loss_fn = loss_fns.get_loss_fn(model_name)
    optimizer = loss_fns.get_optimizer(model.parameters(), args.optimizer, args.lr, args.momentum, args.weight_decay)
    best_val_f1 = 0

    for i in range(args.epochs if not args.eval_on_test else 1):
        print('Epoch: {}'.format(i))

        vis_logger.reset_epoch_data()

        for split in ['train', 'val'] if not args.eval_on_test else ['test']:
            train_data = dataloaders.get_subset(split)

            train_loader = get_train_loader('standard', train_data, args.batch_size)
            model.train() if split == ['train'] else model.eval()
            nclass = len(CM_LABELS[args.country]) + 1
            # for inputs, targets, cloudmasks, hres_inputs in tqdm(dl):
            for inputs, targets, cloudmasks in tqdm(train_loader):

                targets[targets > 4] = 0
                targets = F.one_hot(targets.to(torch.int64), num_classes=nclass)
                mask = torch.arange(1, 5)  # tensor([1, 2, 3, 4])

                targets = torch.index_select(targets, 3, mask)

                targets = targets.permute(0, 3, 1, 2)
                cloudmasks = None

                with torch.set_grad_enabled(True):
                    if not args.var_length:
                        # inputs.to(args.device)
                        for a in inputs:
                            inputs[a].to(args.device)

                        # if hres_inputs is not None: hres_inputs.to(args.device)
                    else:
                        for sat in inputs:
                            if "length" not in sat:
                                inputs[sat].to(args.device)
                    targets.to(args.device)

                    # ## For 2nd command(unet3d) uncommnet later commands
                    inputs=torch.cat( (inputs['s1'],inputs['s2'],inputs['planet']), dim=1)
                    inputs=inputs.permute(0,1,4,2,3)  #torch.Size([2, 17, 64, 64, 256]) After permute torch.Size([2, 17, 256, 64, 64])
                    inputs=inputs.float()
                    inputs=inputs.cuda()

                    # For 3rd command() uncommnet later commands
                    # for a in inputs:
                    #     inputs[a] = inputs[a].permute(0, 4, 1, 2, 3)  ##  batch, timestamps, bands, rows, columns
                    #     inputs[a] = inputs[a].float().cuda()

                    # # ## For 4th command() uncommnet later commands
                    # inputs=torch.cat( (inputs['s1'],inputs['s2'],inputs['planet']), dim=1)
                    # inputs=inputs.permute(0,4,1,2,3)
                    # inputs=inputs.float().cuda()

                    preds = model(inputs)
                    predictions.append(preds)
                    loss, cm_cur, total_correct, num_pixels, confidence = evaluate(model_name, preds, targets,
                                                                                   args.country, loss_fn=loss_fn,
                                                                                   reduction="sum",
                                                                                   loss_weight=args.loss_weight,
                                                                                   weight_scale=args.weight_scale,
                                                                                   gamma=args.gamma)

                    if split == 'train' and loss is not None:  # TODO: not sure if we need this check?
                        # If there are valid pixels, update weights
                        optimizer.zero_grad()
                        # with autograd.detect_anomaly():
                        loss.backward()
                        if args.clip_val:
                            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                        optimizer.step()
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        gradnorm = total_norm ** (1. / 2)
                        # gradnorm = torch.norm(list(model.parameters())[0].grad).detach().cpu() / torch.prod(torch.tensor(list(model.parameters())[0].shape), dtype=torch.float32)

                        # vis_logger.update_progress('train', 'gradnorm', gradnorm)
                        loss = loss.cpu()
                        with torch.no_grad():
                            vis_logger.update_progress('train', 'gradnorm', gradnorm)

                    if cm_cur is not None:  # TODO: not sure if we need this check?
                        # If there are valid pixels, update metrics
                        with torch.no_grad():
                            vis_logger.update_epoch_all(split, cm_cur, loss, total_correct, num_pixels)
                        # vis_logger.update_epoch_all(split, cm_cur, loss, total_correct, num_pixels)

                # with torch.no_grad():
                #     vis_logger.record_batch(inputs, cloudmasks, targets.float(), preds, confidence,
                #                         NUM_CLASSES[args.country], split,
                #                         args.include_doy, args.use_s1, args.use_s2,
                #                         model_name, args.time_slice, var_length=args.var_length)

                # del loss  # adding this as the gpu usage was increasing with time
                # del preds

            if split in ['test']:
                vis_logger.record_epoch(split, i, args.country, save=False,
                                        save_dir=os.path.join(args.save_dir, args.name + "_best_dir"))
            else:
                vis_logger.record_epoch(split, i, args.country)

            if split == 'val':
                val_f1 = metrics.get_f1score(vis_logger.epoch_data['val_cm'], avg=True)
                print("val at epoch=", i, val_f1)

                if val_f1 > best_val_f1:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.name + "_best"))
                    best_val_f1 = val_f1
                    print("best till now saved at epoch", i, args.name + "_best")

                    # """  # commenting the next lines to check now if it is working or not
                    if args.save_best:
                        # TODO: Ideally, this would save any batch except the last one so that the saved images
                        #  are not only the remainder from the last batch
                        vis_logger.record_batch(inputs, cloudmasks, targets.float(), preds, confidence,
                                                NUM_CLASSES[args.country], split,
                                                args.include_doy, args.use_s1, args.use_s2,
                                                model_name, args.time_slice, save=True, var_length=args.var_length,
                                                save_dir=os.path.join(args.save_dir, args.name + "_best_dir"))

                        vis_logger.record_epoch(split, i, args.country, save=True,
                                                save_dir=os.path.join(args.save_dir, args.name + "_best_dir"))

                        vis_logger.record_epoch('train', i, args.country, save=True,
                                                save_dir=os.path.join(args.save_dir, args.name + "_best_dir"))

    return predictions


def predict(model, model_name, args=None, dataloaders=None, X=None, y=None):
    """ Trains the model on the inputs

    Args:
        model - trainable model
        model_name - (str) name of the model
        args - (argparse object) args parsed in from main; used only for DL models
        dataloaders - (dict of dataloaders) used only for DL models
        X - (npy arr) data for non-dl models
        y - (npy arr) labels for non-dl models
    """
    if dataloaders is None and model_name in DL_MODELS: raise ValueError("DATA GENERATOR IS NONE")
    if args is None and model_name in DL_MODELS: raise ValueError("Args is NONE")

    if model_name in NON_DL_MODELS:
        predictions = train_non_dl_model(model, model_name, dataloaders, args, X, y)
    elif model_name in DL_MODELS:
        predictions = predict_dl_model(model, model_name, dataloaders, args)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return predictions


def main(args):
    if args.seed is not None:
        if args.device == 'cuda':
            use_cuda = True
        elif args.device == 'cpu':
            use_cuda = False
        util.random_seed(seed_value=args.seed, use_cuda=use_cuda)

    # load in data generator

    dataset = get_dataset(dataset='africa_crop_type_mapping', split_scheme="cauvery", resize_planet=True,
                          normalize=True, calculate_bands=True, root_dir=args.path_to_cauvery_images)

    dataloaders = dataset

    # load in model
    model = croptype_models.get_model(**vars(args))
    if args.model_name in DL_MODELS:
        print('Total trainable model parameters: {}'.format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
        # model.load_state_dict(torch.load("PATH/TO/MODEL"))

    if args.model_name in DL_MODELS and args.device == 'cuda' and torch.cuda.is_available():
        model.to(args.device)

    if args.name is None:
        args.name = str(datetime.datetime.now()) + "_" + args.model_name

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print("Starting to predict")
    return predict(model, args.model_name, args, dataloaders=dataloaders)


if __name__ == "__main__":
    # parse args
    parser = util.get_train_parser()

    main(parser.parse_args())

