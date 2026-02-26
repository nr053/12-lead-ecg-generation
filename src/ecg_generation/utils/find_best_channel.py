"""This is to find the best precordial channel to use as input"""

import wfdb
import glob
from tqdm import tqdm
import statistics
from ecg_generation.utils.unet import UNet
from ecg_generation.loading.incart_dataset import INCART_recording
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



#all recordings in the INCART data set
filenames = glob.glob("/home/rose/Cortrium/12-lead-ecg-generation/data/INCART/" + '/**/*.dat', recursive=True) #find all the filenames with .dat file extension

input_lead_combos = [
    ['I', 'II', 'V1'],
    ['I', 'II', 'V2'],
    ['I', 'II', 'V3'],
    ['I', 'II', 'V4'],
    ['I', 'II', 'V5'],
    ['I', 'II', 'V6']
]

output_lead_combos = [
    ['V2', 'V3', 'V4', 'V5', 'V6'],
    ['V1', 'V3', 'V4', 'V5', 'V6'],
    ['V1', 'V2', 'V4', 'V5', 'V6'],
    ['V1', 'V2', 'V3', 'V5', 'V6'],
    ['V1', 'V2', 'V3', 'V4', 'V6'],
    ['V1', 'V2', 'V3', 'V4', 'V5'],
]

best_channel_dict = dict()
best_channel_dict["input_combo"] = []
best_channel_dict["recording"] = []
best_channel_dict["mse"] = []
best_channel_dict["r2"] = []
best_channel_dict["pc"] = []


for input_combo, output_combo in zip(input_lead_combos, output_lead_combos):
    config["input_leads"] = input_combo
    config["predicted_leads"] = output_combo

    for file in tqdm(filenames, desc="creating dataset"):
        tag = file.split("/")[-1].removesuffix(".dat")

        #load dataset into custom data class
        dataset = INCART_recording(file, 
                                config,
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
        train = Train_class(directory="find_best_channel/single_file", model_tag=tag, train_set=subset_train, val_set=subset_val)
        model_name = train.train_loop(n_epochs=300)

        #test
        test = Test_class(model_tag=model_name, dataset=dataset)
        test.predict()
        df_test = dataset.df.loc[subset_test.indices] #only calculate MSE on test set
        mse = df_test["mse"].mean()
        r2 = df_test["r2"].mean()
        pc = df_test["pc"].mean()
        best_channel_dict["input_combo"].append(input_combo)
        best_channel_dict["recording"].append(tag)
        best_channel_dict["mse"].append(mse)
        best_channel_dict["r2"].append(r2)
        best_channel_dict["pc"].append(pc)



df = pd.DataFrame(best_channel_dict)
