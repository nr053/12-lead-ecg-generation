"""This script evaluates performance of ALL single patient models on ALL patient

    - iterate through the patients one by one
    - load relevant trained model
    - iterate through all patients
        - load relevant recordings into dataset
        - make predictions
        - save MSE and patient name in dataframe
"""


import wfdb
import glob
from tqdm import tqdm
import statistics
from unet import UNet
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
import os 

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#read the file listing patients and associated recordings
with open(config['path_to_data'] + "files/incartdb/1.0.0/files-patients-diagnoses.txt", "r") as f:
    string = f.readlines()

#create dictionary of patients and associated recordings
my_dict = {}
for i in range(len(string)):
    if string[i].startswith('patient'):
        my_dict[string[i].removesuffix("\n")] = string[i+1].removesuffix("\n")

#create list of models
model_list = os.listdir(config['path_to_models'] + 'single_patient_new/')

#init performance dict
results_dict = dict()
results_dict["patient model"] = []
results_dict["test patient"] = []
results_dict["mse"] = []
results_dict["pc"] = []

# iterate through every patient 
#   load relevant model
#   iterate through every patient
#      create dataset with relevant recordings 
#      make predictions on entire set
#      save performance on test_set to performance dict
for trained_patient, r in tqdm(my_dict.items()):
    #select model
    model_name = next((model for model in model_list if trained_patient in model), None)
    if model_name is not None:
        model_path = config['path_to_models'] + "single_patient_new/" + model_name

        for test_patient, recordings in tqdm(my_dict.items()):
            dataset = INCART_patient(recordings.split(), 
                                    config
                                    )

            #set fixed seed
            generator = torch.Generator().manual_seed(42)

            #test
            test = Test_class(model_tag=model_path, dataset=dataset)
            test.predict()
            #calculate performance
            mse = dataset.df["mse"].mean()
            pc = dataset.df["pc"].mean()
            #save
            results_dict["patient model"].append(trained_patient)
            results_dict["test patient"].append(test_patient)
            results_dict["mse"].append(mse)
            results_dict["pc"].append(pc)

df = pd.DataFrame(results_dict)


