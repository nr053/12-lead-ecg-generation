from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch
from statistics import mean
import matplotlib.pyplot as plt
import os
import math
from unet import UNet
from sklearn import preprocessing
import numpy as np
from load_incart import INCART_LOADER
import yaml

config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset = INCART_LOADER(config['path_to_data'], config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

a = np.array([
    [-5,-4,-3,-2,-1,0,-1,-2,-3,-4,-5],
    [-10,-8,-6,-4,-2,-0,-2,-4,-6,-8,-10],
    [0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
])

b = preprocessing.normalize(a, axis=0)
c = preprocessing.normalize(a, axis=1)

import wfdb
import pylab as pylab
import scipy
#import biosppy
import matplotlib.pyplot as plt

path = "/home/rose/Cortrium/12-lead-ecg-generation/data/files/incartdb/1.0.0/I10"

signal = wfdb.rdrecord(path)
annotation = wfdb.rdann(path, extension="atr")

#mat = scipy.io.loadmat(path)

#r_peaks = biosppy.signals.ecg.ASI_segmenter(signal.p_signal[:,6], sampling_rate=500)

def butter(low, high):
    b, a = scipy.signal.butter(5, [low, high], btype='bandpass', fs=500)
    return scipy.signal.lfilter(b, a, signal.p_signal[:,6])

filtered_1 = butter(0.5,150)
filtered_2 = butter(1,120)
filtered_3 = butter(3,100)
filtered_4 = butter(3,70)
filtered_5 = butter(3,50)
filtered_6 = butter(5,50)





fig, axs = plt.subplots(7,1)

axs[0].plot(signal.p_signal[:257,6])
axs[1].plot(filtered_1)
axs[2].plot(filtered_2)
axs[3].plot(filtered_3)
axs[4].plot(filtered_4)
axs[5].plot(filtered_5)
axs[6].plot(filtered_6)

#for peak in r_peaks[0]:
#    plt.axvline(x=peak)


plt.savefig("sandbox.png")