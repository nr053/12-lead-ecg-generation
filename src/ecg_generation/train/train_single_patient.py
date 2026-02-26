import wfdb
import glob
from tqdm import tqdm
import statistics
from ecg_generation.utils.unet import UNet
from ecg_generation.loading.incart_patient import INCART_patient
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

#read the file listing patients and associated recordings
with open("/home/rose/Cortrium/12-lead-ecg-generation/data/INCART/files/incartdb/1.0.0/files-patients-diagnoses.txt", "r") as f:
    string = f.readlines()

#create dictionary of patients and associated recordings
my_dict = {}
for i in range(len(string)):
    if string[i].startswith('patient'):
        my_dict[string[i].removesuffix("\n")] = string[i+1].removesuffix("\n")


results_dict = dict()
results_dict["patient"] = []
results_dict["mse"] = []
results_dict["r2"] = []
results_dict["pc"] = []

#iterate through every patient creating a dataset for each recording and then concatenating them to form 
#one dataset containing every recording from the same patient
for patient, recordings in tqdm(my_dict.items()):
    dataset = INCART_patient(recordings.split(), 
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
    train = Train_class(directory="single_patient", model_tag=patient, train_set=subset_train, val_set=subset_val)
    model_name = train.train_loop(n_epochs=300)

    #test
    test = Test_class(model_tag=model_name, dataset=dataset)
    test.predict()
    df_test = dataset.df.loc[subset_test.indices] #only calculate MSE on test set
    mse = df_test["mse"].mean()
    r2 = df_test["r2"].mean()
    pc = df_test["pc"].mean()
    results_dict["patient"].append(patient)
    results_dict["mse"].append(mse)
    results_dict["r2"].append(r2)
    results_dict["pc"].append(pc)

df = pd.DataFrame(results_dict)


