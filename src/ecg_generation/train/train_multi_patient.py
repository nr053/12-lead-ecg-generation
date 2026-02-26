import wfdb
import glob
from tqdm import tqdm
import statistics
from ecg_generation.utils.unet import UNet
from ecg_generation.loading.incart_dataset import INCART_Dataset
from ecg_generation.utils.training_utils import Train_class, Test_class
import yaml
import random
import torch
import os
import pandas as pd
from torch.utils.data import random_split
import random
import argparse

#arguments
parser = argparse.ArgumentParser(
                    prog='predict',
                    description='generate 12-lead ECG on a sample file using a specific model',
                    epilog='Text at the bottom of help')
parser.add_argument(
    '-b', 
    '--balance',
    action="store_true",
    help="Enable balancing of dataset"
)
parser.add_argument(
    '-c', 
    '--centre',
    action="store_true",
    help="enable centering of beats in each sample"
)
args = parser.parse_args()


if args.balance:
    tag = "balanced"
else:
    tag = "un_balanced"

if args.centre:
    tag = tag + '_centred'
else:
    tag = tag + '_uncentred'
#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset_train = INCART_Dataset(config["path_to_data"] + "files/incartdb/1.0.0/train/", config, balance_data=args.balance, centre_sample=args.centre)
dataset_val = INCART_Dataset(config["path_to_data"] + "files/incartdb/1.0.0/val/", config)
dataset_test = INCART_Dataset(config["path_to_data"] + "files/incartdb/1.0.0/test/", config)

#set fixed seed
generator = torch.Generator().manual_seed(42)

#train
train = Train_class(directory="multi_patient", model_tag=tag, train_set=dataset_train, val_set=dataset_val)
model_name = train.train_loop(n_epochs=300)

#test
test = Test_class(model_tag=model_name, dataset=dataset_test)
test.predict()

print(f"mean MSE: {dataset_test.df["mse"].mean()}")
print(f"min MSE: {dataset_test.df["mse"].min()}")
print(f"max MSE: {dataset_test.df["mse"].max()}")
print(f"std dev MSE: {dataset_test.df["mse"].std()}")

