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

class C3_dataset(INCART_Dataset):
    def __init__(self, path_cardioline, path_cortrium, config):
        self.config = config
        #self.fs = fs
        #fourier transform stuff
        w = scipy.signal.windows.hann(config['window_size'])
        self.SFT = scipy.signal.ShortTimeFFT(w, hop=config['hop'], fs=config['s_freq'])
        #preprocessing
        self.scaler = preprocessing.MinMaxScaler()
        #dataframe
        self.df = self.create_data_set(path_cardioline, path_cortrium)

        channel_map = {
            'I': 0,
            'II': 1,
            'III': 2,
            'aVR': 3,
            'aVF': 4,
            'aVL': 5,
            'V1': 6,
            'V2': 7,
            'V3': 8,
            'V4': 9,
            'V5': 10,
            'V6': 11,
            'ECG1': 12,
            'ECG2': 13,
            'ECG3': 14,
        }
        
        self.input_lead_names = ['ECG1', 'ECG2', 'ECG3']
        self.input_lead_idx = [channel_map[input_lead_name] for input_lead_name in self.input_lead_names]
        self.input_stft_idx = [idx for lead in self.input_lead_idx for idx in (lead*2, (lead*2)+1)]

        self.output_lead_names = [
            'I', 
            'II', 
            'III',
            'aVR',
            'aVF',
            'aVL', 
            'V1', 
            'V2', 
            'V3', 
            'V4', 
            'V5', 
            'V6', 
            ]
        self.output_lead_idx = [channel_map[output_lead_name] for output_lead_name in self.output_lead_names]


    def create_data_set(self, path_cardioline, path_cortrium):
        filenames_cardioline = [
            path_cardioline + '59.dat',
            path_cardioline + '60.dat',
            path_cardioline + '61.dat',
            path_cardioline + '62.dat',
            path_cardioline + '63.dat',
            path_cardioline + '64.dat',
            path_cardioline + '65.dat',
            path_cardioline + '66.dat',
            path_cardioline + '67.dat',
        ]

        sample_start_times = [
            337835,
            350170,
            363598,
            376663,
            388681,
            402894,
            415242,
            427675,
            442650
        ]

        raw = mne.io.read_raw_edf(path_cortrium, preload=True)
        data_c3 = raw.get_data()
        
        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['ecg_original'] = []
        set_dict['stft'] = []
        set_dict['stft_pred'] = []
        set_dict['ecg_pred'] = []
        set_dict['annotation'] = []
        set_dict['recording'] = []

        for file, start_time in tqdm(zip(filenames_cardioline, sample_start_times), desc="processing beats"):
            tqdm.write(f"loading file: {file}")
            strips = self.load_file(file.removesuffix(".dat"), data_c3, start_time)
            for ecg_strip, annotation_strip, ecg_strip_transformed, ecg_strip_original in zip(*strips):
                set_dict['ecg'].append(ecg_strip)
                set_dict['ecg_original'].append(ecg_strip_original)
                set_dict['stft'].append(ecg_strip_transformed)
                set_dict['stft_pred'].append([])
                set_dict['ecg_pred'].append([])
                set_dict['annotation'].append(annotation_strip)
                set_dict['recording'].append(file.split("/")[-1].removesuffix(".dat"))
        df = pd.DataFrame(set_dict)

        return df

    def load_file(self, filename, data_c3, start_time):
        """The procedure to handle the cortrium X cardioline dataset
        
        input: path to file (string)
        ouput: array with signal data and accompanying annotation label
                possibly with metadata of patient / record ????

        """
        #load the file into a WFDB record object
        signal = wfdb.rdrecord(filename)

        #get the appropriate c3 chunk
        data_c3 = data_c3[:3, start_time:start_time+(256*10)]
        data_c3 = np.swapaxes(data_c3, 0, 1)

        #resample the signal
        signal_resampled = resample_poly(signal.p_signal, 256, 500, axis=0)
        
        #stack 12-lead and 3-lead data
        signal_stacked = np.concatenate([signal_resampled, data_c3],axis=1)
        
        #filter the signal
        b, a = scipy.signal.butter(5, [3, 50], btype='bandpass', fs=self.config['s_freq'])
        signal_filtered = scipy.signal.lfilter(b, a, signal_stacked)
        #signal_filtered = signal.p_signal #currently no filtering

        #init lists to build set dict
        ecg_strips = []
        annotation_strips = []
        ecg_strips_transformed = []
        ecg_strips_original = []

        #input snippets are chopped regardless of beat position
        index = 0
        while index <= signal_filtered.shape[0] - 256: #only include strips of 256 length
            ecg_strip = signal_filtered[index:index+256,:]
            ecg_strip = np.swapaxes(ecg_strip, 0, 1)

            #normalise
            ecg_normalised_channels = []
            for channel in ecg_strip: 
                ecg_normalised_channels.append(self.scaler.fit_transform(channel.reshape(-1,1)))
            ecg_strip_normalised = np.array(ecg_normalised_channels)[:,:,-1]

            #find beat annotation
            beat_annotations = ""
            #for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
            #    if index <= beat_position <= index+256:
            #        beat_annotations += beat_annotation    
            
            #preprocessing step (fourier transform)
            ecg_strip_transformed = self.preprocess(ecg_strip_normalised)
            
            #append strip
            ecg_strips.append(ecg_strip_normalised)
            annotation_strips.append(beat_annotations)
            ecg_strips_transformed.append(ecg_strip_transformed)
            ecg_strips_original.append(ecg_strip)


            index += 256

        return ecg_strips, annotation_strips, ecg_strips_transformed, ecg_strips_original

        

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        input = sample.stft[self.input_stft_idx,:]
        output = sample.stft[:24,:]

        return {'x':input, 'y':output}

