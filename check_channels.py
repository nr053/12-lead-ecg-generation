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

for file in filenames:

    #--- load file ---
    signal = wfdb.rdrecord(file.removesuffix(".dat"))
    # annotation = wfdb.rdann(path, extension="atr")

    print(signal.sig_name)

