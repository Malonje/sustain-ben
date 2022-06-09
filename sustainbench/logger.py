import os
import wandb
import datetime
import pandas as pd


def init(project="dilineation", reinit=True):
    # dataset_type = opts['dataset_type']
    wandb.init(project=project, entity='ai4sg', reinit=reinit)
    # if run_name:
    #     wandb.run.name = run_name# + "(" + wandb.run.name + ")"
    return wandb.run.name


def log(summary, step=None):
    if step:
        wandb.log(summary)
    else:
        wandb.log(summary)

