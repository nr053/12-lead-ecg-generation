import wfdb
import glob
from tqdm import tqdm
import statistics
from unet import UNet
from load_incart import INCART_Dataset
import yaml
import random
import torch
import os
from training_utils import Train_class, Test_class
from unet import UNet
import pandas as pd
from torch.utils.data import random_split
import random

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset = INCART_Dataset(config["path_to_data"] + "files/incartdb/1.0.0/train/I03.dat", config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

# #set fixed seed
# generator = torch.Generator().manual_seed(42)

# #make train/val/test split 50%/30%/20% 
# train_size = int(len(dataset)*0.5)
# test_size = int(len(dataset)*0.2)
# val_size = len(dataset) - (train_size + test_size)
# subset_train, subset_val, subset_test = random_split(dataset, [train_size, val_size, test_size], generator=generator)


# df_train = dataset.df.loc[subset_train.indices] #only calculate MSE on validation set (should obvs be test set)
# df_val = dataset.df.loc[subset_val.indices] #only calculate MSE on validation set (should obvs be test set)
# df_test = dataset.df.loc[subset_test.indices] #only calculate MSE on validation set (should obvs be test set)

# results_dict = dict()
# results_dict["file"] = []
# results_dict["mse"] = []


# #train
# train = Train_class(model_tag=tag, train_set=subset_train, val_set=subset_val)
# model_name = train.train_loop(n_epochs=200)

# #test
# test = Test_class(model_tag=model_name, dataset=dataset)
# test.predict()
# df_test = dataset.df.loc[subset_val.indices] #only calculate MSE on validation set (should obvs be test set)
# mse = df_test["mse"].mean()
# results_dict["file"].append(tag)
# results_dict["mse"].append(mse)

# df = pd.DataFrame(results_dict)

