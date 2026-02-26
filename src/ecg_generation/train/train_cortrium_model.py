import wfdb
import glob
from tqdm import tqdm
import statistics
from ecg_generation.utils.unet import UNet
from ecg_generation.loading.cortrium_patient import C3_dataset
from ecg_generation.utils.training_utils import Train_class, Test_class
import yaml
import random
import torch
import os
import pandas as pd
from torch.utils.data import random_split
import random

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

dataset = C3_dataset(
    "/home/rose/Cortrium/12-lead-ecg-generation/data/SCP/",
    "/home/rose/Cortrium/12-lead-ecg-generation/data/C3/brandon.EDF",
    config
    )


#set fixed seed
generator = torch.Generator().manual_seed(42)

#make train/val split (currently no test split)
set_size = len(dataset)
train_size = int(set_size*0.7) #70% of data
val_size = int(set_size*0.2) #20% of data
test_size = set_size - (train_size + val_size) #remaining 10% of data
subset_train, subset_val, subset_test = random_split(dataset, [train_size, val_size, test_size], generator=generator)

#train
train = Train_class(directory="cortrium", model_tag="cortrium", train_set=subset_train, val_set=subset_val)
model_name = train.train_loop(n_epochs=300000)


#test
test = Test_class(model_tag=model_name, dataset=dataset)
test.predict()
df_test = dataset.df.loc[subset_test.indices] #only calculate MSE on test set
mse = df_test["mse"].mean()
r2 = df_test["r2"].mean()
pc = df_test["pc"].mean()

