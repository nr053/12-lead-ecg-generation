"""This script evaluates performance on a multi patient model

    - load relevant recordings into test set
    - load relevant trained model
    - make predictions
    - print peformance
    
"""


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
parser.add_argument('model')           # positional argument
args = parser.parse_args()

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset_test = INCART_Dataset(config["path_to_data"] + "files/incartdb/1.0.0/test/", config)

#test
test = Test_class(model_tag=config["path_to_models"] + f"multi_patient/{args.model}", dataset=dataset_test)
test.predict()

print(f"mean MSE: {dataset_test.df["mse"].mean()}")
print(f"min MSE: {dataset_test.df["mse"].min()}")
print(f"max MSE: {dataset_test.df["mse"].max()}")
print(f"std dev MSE: {dataset_test.df["mse"].std()}")

