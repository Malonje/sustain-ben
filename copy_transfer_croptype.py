from cmath import nan
import os
import copy_loss_fns
import croptype_models
import datetime
import torch
# import datasets
import metrics
import util
import numpy as np
import pickle
from sustainbench import logger

from torch import autograd

from constants import *
from tqdm import tqdm
from torch import autograd
import visualize

from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import torch.nn.functional as F


def evaluate_split(model, model_name, split_loader, device, loss_weight, weight_scale, gamma, num_classes, country,
                   var_length):
    total_loss = 0
    total_pixels = 0
    total_cm = np.zeros((num_classes, num_classes)).astype(int)
    loss_fn = copy_loss_fns.get_loss_fn(model_name)
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

            accuracy = metrics.get_accuracy(model_name, preds, labels, reduction=reduction)
            return loss, cm, accuracy, confidence
        elif reduction == "sum":
            loss, confidence, _ = loss_fn(labels, preds, reduction, country, loss_weight, weight_scale)
            total_correct, num_pixels = metrics.get_accuracy(model_name, preds, labels, reduction=reduction)
            return loss, cm, total_correct, num_pixels, confidence
        else:
            raise ValueError(f"reduction: `{reduction}` not supported")


def train_dl_model(model, model_name, dataloaders, args):
    config = {
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'use_l8': args.use_l8,
        'use_s1': args.use_s1,
        'use_s2': args.use_s2,
        'use_planet': args.use_planet,
    }
    run_name = logger.init(project='crop_type_mapping_v3', reinit=True, run_name=args.run_name, config=config)
    # splits = ['train', 'val'] if not args.eval_on_test else ['test']
    sat_names = ""
    if args.use_s1:
        sat_names += "S1"
    if args.use_s2:
        sat_names += "S2"
    if args.use_l8:
        sat_names += "L8"
    if args.use_planet:
        sat_names += "planet"





    if args.clip_val:
        clip_val = sum(p.numel() for p in model.parameters() if p.requires_grad) // 20000
        print('clip value: ', clip_val)

    loss_fn = copy_loss_fns.get_loss_fn(model_name)
    optimizer = copy_loss_fns.get_optimizer(model.parameters(), args.optimizer, args.lr, args.momentum, args.weight_decay)
    best_val = 0

    for i in range(args.epochs if not args.eval_on_test else 1):
        print('Epoch: {}'.format(i))

        for split in ['train', 'val'] if not args.eval_on_test else ['val', 'test']:
            correct_pixels = 0
            total_pixels = 0
            losses = []
            train_data = dataloaders.get_subset(split)

            if split == 'train':
                train_loader = get_train_loader('standard', train_data, args.batch_size)
                model.train()
            else:
                train_loader = get_eval_loader('standard', train_data, args.batch_size)
                model.eval()

            nclass = 184 + 1
            # for inputs, targets, cloudmasks, hres_inputs in tqdm(dl):
            for inputs, targets in tqdm(train_loader):
                targets = F.one_hot(targets.to(torch.int64), num_classes=nclass)
                mask = torch.arange(1, 3)  # tensor([1, 2, 3, 4])

                targets = torch.index_select(targets, 3, mask)

                targets = targets.permute(0, 3, 1, 2)
                # cloudmasks = None

                with torch.set_grad_enabled(True):
                    if not args.var_length:
                        for a in inputs:
                            inputs[a].to(args.device)
                    else:
                        for sat in inputs:
                            if "length" not in sat:
                                inputs[sat].to(args.device)
                    targets.to(args.device)

                    temp_inputs = None
                    if args.use_s1:
                        temp_inputs = inputs['s1']
                    if args.use_s2:
                        if temp_inputs is None:
                            temp_inputs = inputs['s2']
                        else:
                            temp_inputs = torch.cat((temp_inputs, inputs['s2']), dim=1)
                    if args.use_l8:
                        if temp_inputs is None:
                            temp_inputs = inputs['l8']
                        else:
                            temp_inputs = torch.cat((temp_inputs, inputs['l8']), dim=1)
                    if args.use_planet:
                        if temp_inputs is None:
                            temp_inputs = inputs['planet']
                        else:
                            temp_inputs = torch.cat((temp_inputs, inputs['planet']), dim=1)

                    inputs = temp_inputs
                    # inputs = torch.cat((inputs['s1'], inputs['s2'], inputs['planet']), dim=1)
                    # print(inputs.shape)
                    inputs = inputs.permute(0, 1, 4, 2, 3)  # torch.Size([2, 17, 64, 64, 256]) After permute torch.Size([2, 17, 256, 64, 64])
                    inputs = inputs.float()
                    inputs = inputs.cuda()

                    preds = model(inputs)
                    loss, cm_cur, total_correct, num_pixels, confidence = evaluate(model_name, preds, targets,
                                                                                   args.country, loss_fn=loss_fn,
                                                                                   reduction="sum",
                                                                                   loss_weight=args.loss_weight,
                                                                                   weight_scale=args.weight_scale,
                                                                                   gamma=args.gamma)
                    correct_pixels += total_correct
                    total_pixels += num_pixels
                    losses.append(loss)

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

            accuracy = correct_pixels / total_pixels

            if split == 'test':
                print(f"[Test] #Correct: {correct_pixels}, #Pixels {total_pixels}, Accuracy: {accuracy}, "
                      f"Loss: {sum(losses) / len(losses)}")
            else:
                if split == 'val':
                    logger.log({
                        f"Validation Accuracy": accuracy,
                        f"Validation Loss": sum(losses) / len(losses),
                        "X-Axis": i,
                    })
                    print(f"[Validation] #Correct: {correct_pixels}, #Pixels {total_pixels}, Accuracy: {accuracy}")
                    if best_val < accuracy and not args.eval_on_test:
                        best_val = accuracy
                        torch.save(model.state_dict(), f"../model_weights/{run_name}.pth.tar")
                else:
                    logger.log({
                        f"Train Accuracy": accuracy,
                        f"Train Loss": sum(losses) / len(losses),
                        "X-Axis": i
                    })
                    print(f"[Train] #Correct: {correct_pixels}, #Pixels {total_pixels}, Accuracy: {accuracy}")

    if args.use_testing:
        model.load_state_dict(torch.load(f"../model_weights/{run_name}.pth.tar"))
        for split in  ['val', 'test']:
                correct_pixels = 0
                total_pixels = 0
                train_data = dataloaders.get_subset(split)

                if split == 'train':
                    train_loader = get_train_loader('standard', train_data, args.batch_size)
                    model.train()
                else:
                    train_loader = get_eval_loader('standard', train_data, args.batch_size)
                    model.eval()

                nclass = 184 + 1
                # for inputs, targets, cloudmasks, hres_inputs in tqdm(dl):
                for inputs, targets in tqdm(train_loader):
                    targets = F.one_hot(targets.to(torch.int64), num_classes=nclass)
                    mask = torch.arange(1, 3)  # tensor([1, 2, 3, 4])

                    targets = torch.index_select(targets, 3, mask)

                    targets = targets.permute(0, 3, 1, 2)
                    # cloudmasks = None

                    with torch.set_grad_enabled(True):
                        if not args.var_length:
                            for a in inputs:
                                inputs[a].to(args.device)
                        else:
                            for sat in inputs:
                                if "length" not in sat:
                                    inputs[sat].to(args.device)
                        targets.to(args.device)

                        temp_inputs = None
                        if args.use_s1:
                            temp_inputs = inputs['s1']
                        if args.use_s2:
                            if temp_inputs is None:
                                temp_inputs = inputs['s2']
                            else:
                                temp_inputs = torch.cat((temp_inputs, inputs['s2']), dim=1)
                        if args.use_l8:
                            if temp_inputs is None:
                                temp_inputs = inputs['l8']
                            else:
                                temp_inputs = torch.cat((temp_inputs, inputs['l8']), dim=1)
                        if args.use_planet:
                            if temp_inputs is None:
                                temp_inputs = inputs['planet']
                            else:
                                temp_inputs = torch.cat((temp_inputs, inputs['planet']), dim=1)

                        inputs = temp_inputs
                        # inputs = torch.cat((inputs['s1'], inputs['s2'], inputs['planet']), dim=1)
                        # print(inputs.shape)
                        inputs = inputs.permute(0, 1, 4, 2, 3)  # torch.Size([2, 17, 64, 64, 256]) After permute torch.Size([2, 17, 256, 64, 64])
                        inputs = inputs.float()
                        inputs = inputs.cuda()

                        preds = model(inputs)
                        loss, cm_cur, total_correct, num_pixels, confidence = evaluate(model_name, preds, targets,
                                                                                       args.country, loss_fn=loss_fn,
                                                                                       reduction="sum",
                                                                                       loss_weight=args.loss_weight,
                                                                                       weight_scale=args.weight_scale,
                                                                                       gamma=args.gamma)
                        correct_pixels += total_correct
                        total_pixels += num_pixels

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

                accuracy = correct_pixels / total_pixels

                if split == 'test'  :
                    logger.log({
                            f"Test Accuracy": accuracy,
                        })
                    print(f"[Test] #Correct: {correct_pixels}, #Pixels {total_pixels}, Accuracy: {accuracy}")
                else:
                    if split == 'val':
                        logger.log({
                            f"Validation Accuracy": accuracy,
                            "X-Axis": i,
                        })
                        print(f"[Validation] #Correct: {correct_pixels}, #Pixels {total_pixels}, Accuracy: {accuracy}")
                        if best_val < accuracy and not args.eval_on_test:
                            best_val = accuracy
                            torch.save(model.state_dict(), f"../model_weights/{run_name}.pth.tar")
                    else:
                        logger.log({
                            f"Train Accuracy": accuracy,
                            "X-Axis": i
                        })
                        print(f"[Train] #Correct: {correct_pixels}, #Pixels {total_pixels}, Accuracy: {accuracy}")

def train(model, model_name, args=None, dataloaders=None, X=None, y=None):
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
        train_non_dl_model(model, model_name, dataloaders, args, X, y)
    elif model_name in DL_MODELS:
        train_dl_model(model, model_name, dataloaders, args)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


def main(args):
    if args.seed is not None:
        if args.device == 'cuda':
            use_cuda = True
        elif args.device == 'cpu':
            use_cuda = False
        util.random_seed(seed_value=args.seed, use_cuda=use_cuda)
    l8_bands = [0,1,2,3,4,5,6,7]
    s1_bands = [0,1,2]
    s2_bands = [0,1,2,3,4,5,6,7,8,9]
    ps_bands = [0,1,2,3]
    if args.l8_bands is not None:
        l8_bands=args.l8_bands[1:-1]
        l8_bands=l8_bands.split(',')
        l8_bands=[int(x) for x in l8_bands]
    if args.s1_bands is not None:
        s1_bands=args.s1_bands[1:-1]
        s1_bands=s1_bands.split(',')
        s1_bands=[int(x) for x in s1_bands]
    if args.s2_bands is not None:
        s2_bands=args.s2_bands[1:-1]
        s2_bands=s2_bands.split(',')
        s2_bands=[int(x) for x in s2_bands]
    if args.ps_bands is not None:
        ps_bands=args.ps_bands[1:-1]
        ps_bands=ps_bands.split(',')
        ps_bands=[int(x) for x in ps_bands]

    # load in data generator

    sat_names = ""
    if args.use_s1:
        sat_names += "S1"
    if args.use_s2:
        sat_names += "S2"
    if args.use_l8:
        sat_names += "L8"
    if args.use_planet:
        sat_names += "planet"

    img_dimension = (32,32)
    truth_mask = 10
    if 'planet' in sat_names:
        truth_mask = 3
        img_dimension = (108,108)
    elif sat_names == 'L8':
        truth_mask = 30
        img_dimension = (12,12)
    dataset = get_dataset(dataset='africa_crop_type_mapping', split_scheme="cauvery", resize_planet=True,
                          normalize=True, calculate_bands=True, root_dir=args.path_to_cauvery_images,
                          l8_bands=l8_bands, s1_bands=s1_bands, s2_bands=s2_bands, ps_bands=ps_bands, truth_mask=truth_mask, img_dim=img_dimension)

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

    print("Starting to train")
    # train model
    train(model, args.model_name, args, dataloaders=dataloaders)
    # print("\n\nargs.save_dir= ",args.save_dir)
    # print(args.name)
    # evaluate model

    # save model
    # if args.model_name in DL_MODELS:
    #     torch.save(model.state_dict(), os.path.join(args.save_dir, args.name))
    #     print("MODEL SAVED")


if __name__ == "__main__":
    # parse args
    parser = util.get_train_parser()

    main(parser.parse_args())
