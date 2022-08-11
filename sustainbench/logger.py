import os
import wandb
import datetime
import pandas as pd


def init(project="dilineation", reinit=True, run_name=''):
    # dataset_type = opts['dataset_type']
    wandb.init(project=project, entity='wacv23', reinit=reinit)
    wandb.run.name += ' [' + run_name + ']'
    return wandb.run.name


def log(summary, step=None):
    if step:
        wandb.log(summary)
    else:
        wandb.log(summary)

