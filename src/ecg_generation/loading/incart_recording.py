from ecg_generation.loading.incart_dataset import INCART_Dataset
import wfdb
import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from ecg_generation.utils.unet import UNet
import torch
import torch.utils.data as data
import yaml
from sklearn import preprocessing
import time
from torch.utils.data import Subset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from scipy.signal import resample_poly
import mne

class INCART_recording(INCART_Dataset):
    """This data class is a variation of the INCART_Dataset modified to handle a single recording"""
    def create_data_set(self, balance_data=False, centre_sample=False): #balance dataset is currently not supported in this child class
        file = self.path

        #init placeholder dict
        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['stft'] = []
        set_dict['stft_pred'] = []
        set_dict['ecg_pred'] = []
        set_dict['annotation'] = []
        set_dict['recording'] = []

        #create set dict
        ecg_strips, annotation_strips, ecg_strips_transformed = self.load_file(file.removesuffix(".dat"))
        for ecg_strip, annotation_strip, ecg_strip_transformed in zip(ecg_strips, annotation_strips, ecg_strips_transformed):
            set_dict['ecg'].append(ecg_strip)
            set_dict['stft'].append(ecg_strip_transformed)
            set_dict['stft_pred'].append([])
            set_dict['ecg_pred'].append([])
            set_dict['annotation'].append(annotation_strip)
            set_dict['recording'].append(file.split("/")[-1].removesuffix(".dat"))
        df = pd.DataFrame(set_dict)

        return df