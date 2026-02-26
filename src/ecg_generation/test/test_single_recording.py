"""This script evaluates performance on ALL single recording models

    - iterate through the recordings one by one
    - load relevant dataset
    - load relevant pretrained model
    - save MSE and recording name in dataframe
    
"""


import wfdb
import glob
from tqdm import tqdm
import statistics
from unet import UNet
from ecg_generation.utils.unet import UNet
from ecg_generation.loading.incart_recording import INCART_recording
from ecg_generation.utils.training_utils import Train_class, Test_class
import yaml
import random
import torch
import os
import pandas as pd
from torch.utils.data import random_split
import random
import os 

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#all recordings in the INCART data set
filenames = glob.glob("/home/rose/Cortrium/12-lead-ecg-generation/data/INCART/" + '/**/*.dat', recursive=True) #find all the filenames with .dat file extension

#create list of models
model_list = os.listdir(config['path_to_models'] + 'single_file/')

#init performance dict
results_dict = dict()
results_dict["recording"] = []
results_dict["mse"] = []
results_dict["r2"] = []
results_dict["pc"] = []

results_dict["mse_v1"] = []
results_dict["pc_v1"] = []

results_dict["mse_v2"] = []
results_dict["pc_v2"] = []

results_dict["mse_v4"] = []
results_dict["pc_v4"] = []

results_dict["mse_v5"] = []
results_dict["pc_v5"] = []

results_dict["mse_v6"] = []
results_dict["pc_v6"] = []

# beat_performance_dict = dict()
# beat_performance_dict["type"] = []
# beat_performance_dict["mse"] = []
# beat_performance_dict["pc"] = []

# iterate through every recording
# -* creating a dataset recording 
# *- load relevant model
# *- make predictions on entire set
# *- save performance on test_set to performance dict
for file in tqdm(filenames):
    dataset = INCART_recording(file, 
                            config
                            )

    #set fixed seed
    generator = torch.Generator().manual_seed(42)

    #make train/val/test split
    set_size = len(dataset)
    train_size = int(set_size*0.7) #70% of data
    val_size = int(set_size*0.2) #20% of data
    test_size = set_size - (train_size + val_size) #remaining 10% of data
    subset_train, subset_val, subset_test = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    #select model
    recording_name = file.split("/")[-1].removesuffix(".dat")
    model_name = next((model for model in model_list if recording_name in model), None)
    model_path = config['path_to_models'] + "single_file/" + model_name
    
    print(f"Testing recording: {recording_name}")
    print(f"with model: {model_path}")

    #test
    test = Test_class(model_tag=model_path, dataset=dataset)
    test.predict()
    df_test = dataset.df.loc[subset_test.indices] #only calculate MSE on test set
    
    mse = df_test["mse"].mean()
    r2 = df_test["r2"].mean()
    pc = df_test["pc"].mean()

    mse_v1 = df_test[]

    results_dict["recording"].append(recording_name)
    results_dict["mse"].append(mse)
    results_dict["r2"].append(r2)
    results_dict["pc"].append(pc)

#    beat_performance_dict["type"].extend(df_test["annotation"].values.tolist())
#    beat_performance_dict["mse"].extend(df_test["mse"].values.tolist())
#    beat_performance_dict["pc"].extend(df_test["pc"].values.tolist())

df = pd.DataFrame(results_dict)
#beat_dict = pd.DataFrame(beat_performance_dict)


