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


#all files that the model should use for training and testing
filenames = glob.glob("/home/rose/Cortrium/12-lead-ecg-generation/data/INCART/" + '/**/*.dat', recursive=True) #find all the filenames with .dat file extension
filenames = filenames[:1] #only use 20 files

results_dict = dict()
results_dict["file"] = []
results_dict["mse"] = []

for file in tqdm(filenames, desc="Processing"):
    tqdm.write(f"Handling file: {file}")
    tag = file.split("/")[-1].removesuffix(".dat")

    #load dataset into custom data class
    dataset = INCART_Dataset(file, config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

    #set fixed seed
    generator = torch.Generator().manual_seed(42)

    #make train/val split (currently no test split)
    set_size = len(dataset)
    train_size = int(set_size*0.7) #70% of data
    val_size = int(set_size*0.2) #20% of data
    test_size = set_size - (train_size + val_size) #remaining 10% of data
    subset_train, subset_val, subset_test = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    #train
    train = Train_class(model_tag=tag, train_set=subset_train, val_set=subset_val)
    model_name = train.train_loop(n_epochs=200)

    #test
    test = Test_class(model_tag=model_name, dataset=dataset)
    test.predict()
    df_test = dataset.df.loc[subset_test.indices] #only calculate MSE on validation set (should obvs be test set)
    mse = df_test["mse"].mean()
    results_dict["file"].append(tag)
    results_dict["mse"].append(mse)

df = pd.DataFrame(results_dict)


