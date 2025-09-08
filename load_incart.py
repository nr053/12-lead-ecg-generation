import wfdb
import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from unet import UNet
import torch
import torch.utils.data as data
import yaml
from sklearn import preprocessing


class INCART_LOADER(data.Dataset):
    def __init__(self, path, window_size, hop, fs, channel_map):
        self.path = path
        self.fs = fs
        self.channel_map = channel_map
        #fourier transform stuff
        w = scipy.signal.windows.hann(window_size)
        self.SFT = scipy.signal.ShortTimeFFT(w, hop=hop, fs=self.fs)
        #dataframe
        self.df = self.create_data_set()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        input = sample.stft[
            [self.channel_map[0], 
            self.channel_map[0]+1,
            self.channel_map[1],
            self.channel_map[1+1],
            self.channel_map[2],
            self.channel_map[2]+1]
            ,:]


        return {'x':input, 'y':sample.stft}

    def load_file(self, filename):
        """The procedure to handle a record:
            - create 1s strip with beat in the middle
            - along with beat annotation
        
        input: path to file (string)
        ouput: array with signal data and accompanying annotation label
                possibly with metadata of patient / record ????

        """

        #load the file into a WFDB record object
        signal = wfdb.rdrecord(filename)
        annotation = wfdb.rdann(filename, extension="atr") 

        #filter the signal
        #b, a = scipy.signal.butter(5, [3, 50], btype='bandpass', fs=257)
        #signal_filtered = scipy.signal.lfilter(b, a, signal.p_signal)
        signal_filtered = signal.p_signal #currently no filtering

        #init lists to build set dict
        ecg_strips = []
        annotation_strips = []
        ecg_strips_transformed = []

        #check that the signal sampling freq matches the annotation s_freq and is 257
        if signal.fs != annotation.fs:
            print("reported sampling frequencies do not match")
            return
        elif signal.fs != self.fs:
            print("sampling frequency is not 257")
            return 
        else:
            for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
                #there must be 0.5 seconds of signal either side of the beat position
                if (beat_position > 128) and (beat_position < signal_filtered.shape[0] - 128): 
                    ecg_strip = signal_filtered[beat_position-128:beat_position+128,:]
                    ecg_strip = np.swapaxes(ecg_strip, 0, 1)
                    ecg_strip = preprocessing.normalize(ecg_strip, axis=1) #normalise
                    ecg_strips.append(ecg_strip)
                    annotation_strips.append(beat_annotation)

                    #preprocessing step
                    ecg_strip_transformed = self.preprocess(ecg_strip)
                    ecg_strips_transformed.append(ecg_strip_transformed)



        return ecg_strips, annotation_strips, ecg_strips_transformed, signal, annotation


    def preprocess(self, ecg_strip):
        """preprocess ECG strips with filtering and Fourier transform
        
        Input: 
            ecg_strip (array) : strip of 12 channel ecg
            window_size (int):  size of Hann window for fourier transform
            hop (int): stride used in fourier transform
            fs (int): sampling frequency of ecg
            num_channels (int): number of channels used as input to the model """

        ecg_strip_transformed = []

        for channel in ecg_strip:
            Sx = self.SFT.stft(channel)
            ecg_strip_transformed.append(Sx.real)
            ecg_strip_transformed.append(Sx.imag)
            

        return ecg_strip_transformed



    def plot(self, idx, channel_ids=[0,1,2,3,4,5,6,7,8,9,10,11], reconstructed=False):
        """Plot the ECG channels
        if reconstructed = True, the signal is reconstructed
        from the Fourier Transform"""
        if reconstructed:
            ecg = self.reconstruct_signal(idx)
        else: 
            ecg = self.df.ecg[idx]
        fig, axs = plt.subplots(len(channel_ids),1)
        fig.suptitle(f"Reconstructed: {reconstructed},\n"
                    f"beat type: {self.df.annotation[idx]},\n"
                    f"channel(s): {channel_ids}")
        for i in range(len(channel_ids)):
            axs[i].plot(ecg[channel_ids[i]])

        plt.savefig(f"figures/sample_plots/sample_{idx}_reconstructed_{reconstructed}.png")


    def create_data_set(self):
        filenames = glob.glob(self.path + '/*.dat') #find all the filenames with .dat file extension

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['stft'] = []
        #set_dict['stft_pred'] = []
        #set_dict['ecg_reconstructed'] = []
        set_dict['annotation'] = []


        #TESTING WITH ONLY ONE FILE
        for file in tqdm(filenames[0:1]):
            ecg_strips, annotation_strips, ecg_strips_transformed, signal, annotation = self.load_file(file.removesuffix(".dat"))

            for ecg_strip, annotation_strip, ecg_strip_transformed in zip(ecg_strips, annotation_strips, ecg_strips_transformed):
                set_dict['ecg'].append(ecg_strip)
                set_dict['stft'].append(np.stack(ecg_strip_transformed))
                set_dict['annotation'].append(annotation_strip)
        df = pd.DataFrame(set_dict)

        return df

    def reconstruct_signal(self, idx, prediction=False):
        """Reconstruct a signal from the FFT of the signal"""
        if prediction:
            stft = self.df.prediction[idx]
        else:
            stft = self.df.stft[idx]
        ecg = self.df.ecg[idx]
        reconstructed_signal = []
        for i in range(0,len(stft),2): #increment i in steps of 2
            reconstructed_signal.append(self.SFT.istft(stft[i] + stft[i+1]*1j, k1=ecg.shape[1]))

        return np.array(reconstructed_signal)


