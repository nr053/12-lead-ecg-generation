"""
The dataloader for the INCART dataset: https://physionet.org/content/incartdb/1.0.0/

The parent class INCART_Dataset takes the path to the folder where the dataset is stored and 
recursively iterates through subdirectories finding .dat files to load

The child class INCART_patient takes a list of recording names belonging to the patient in question
e.g. ['I01', 'I02']

Child class INCART_recording takes the path to the recording to be loaded.

"""

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

class INCART_Dataset(data.Dataset):
    def __init__(self, path, config, balance_data=False, centre_sample=False):
        self.config = config
        #self.fs = fs
        #fourier transform stuff
        w = scipy.signal.windows.hann(config['window_size'])
        self.SFT = scipy.signal.ShortTimeFFT(w, hop=config['hop'], fs=config['s_freq'])
        #preprocessing
        self.scaler = preprocessing.MinMaxScaler()
        #dataframe
        self.path = path
        self.df = self.create_data_set(balance_data, centre_sample)

        channel_map = config['channel_map']
        
        self.input_lead_names = config['input_leads']
        self.input_lead_idx = [channel_map[input_lead_name] for input_lead_name in self.input_lead_names]
        self.input_stft_idx = [idx for lead in self.input_lead_idx for idx in (lead*2, (lead*2)+1)]

        self.output_lead_names = config['predicted_leads']
        self.output_lead_idx = [channel_map[output_lead_name] for output_lead_name in self.output_lead_names]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        input = sample.stft[self.input_stft_idx,:]
        output = sample.stft

        return {'x':input, 'y':output}

    def load_file(self, filename, centre_sample=False):
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
        b, a = scipy.signal.butter(5, [3, 50], btype='bandpass', fs=self.config['s_freq'])
        signal_filtered = scipy.signal.lfilter(b, a, signal.p_signal)
        #signal_filtered = signal.p_signal #currently no filtering

        #init lists to build set dict
        ecg_strips = []
        annotation_strips = []
        ecg_strips_transformed = []
        ecg_strips_original = []

        #check that the signal sampling freq matches the annotation s_freq and is 257
        if signal.fs != annotation.fs:
            print("reported sampling frequencies do not match")
            return
        elif signal.fs != self.config['s_freq']:
            print("sampling frequency is not 257")
            return 
        else:
            if centre_sample:
                #each input snippet is centred around a beat
                for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
                    #there must be 0.5 seconds of signal either side of the beat position
                    if (beat_position > 128) and (beat_position < signal_filtered.shape[0] - 128): 
                        ecg_strip = signal_filtered[beat_position-128:beat_position+128,:]
                        ecg_strip = np.swapaxes(ecg_strip, 0, 1)

                        #normalise ECG strip to [0,1] range
                        ecg_normalised_channels = []
                        for channel in ecg_strip: 
                            ecg_normalised_channels.append(self.scaler.fit_transform(channel.reshape(-1,1)))
                        ecg_strip_normalised = np.array(ecg_normalised_channels)[:,:,-1]
                        
                        #preprocessing step (fourier transform)
                        ecg_strip_transformed = self.preprocess(ecg_strip_normalised)

                        #append strips
                        ecg_strips.append(ecg_strip_normalised)
                        annotation_strips.append(beat_annotation)
                        ecg_strips_transformed.append(ecg_strip_transformed)
                        ecg_strips_original.append(ecg_strip)

            else:
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
                    for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
                        if index <= beat_position <= index+256:
                            beat_annotations += beat_annotation    
                    
                    #preprocessing step (fourier transform)
                    ecg_strip_transformed = self.preprocess(ecg_strip_normalised)
                    
                    #append strip
                    ecg_strips.append(ecg_strip_normalised)
                    annotation_strips.append(beat_annotations)
                    ecg_strips_transformed.append(ecg_strip_transformed)
                    ecg_strips_original.append(ecg_strip)


                    index += 256

            


        return ecg_strips, annotation_strips, ecg_strips_transformed, ecg_strips_original


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
            
        return np.stack(ecg_strip_transformed)



    def plot(self, idx, reconstructed=False, predicted=False, original_scale=False):
        """Plot the ECG channels
        if reconstructed = True, the signal is reconstructed
        from the Fourier Transform"""

        filename = self.df.recording[idx]
        tag = ""
        ecg_pred = None
        ecg = self.df.ecg[idx]

        if reconstructed:
            ecg = self.reconstruct_signal(idx)
            tag += "_reconstructed" 
        elif original_scale:
            ecg = self.df.ecg_original[idx]
            ecg_pred = self.df.ecg_pred_original_scale[idx]
            tag += "_originalscale"
        elif predicted:
            ecg_pred = self.df.ecg_pred[idx]
            tag+="_predicted"
        
        #input leads
        fig, axs = plt.subplots(len(self.input_lead_names),1)
        fig.suptitle(f'File {filename} input leads')
        fig.tight_layout()
        for i in range(len(self.input_lead_idx)):
            axs[i].plot(ecg[self.input_lead_idx[i]], label="ground truth")
            axs[i].title.set_text(f'{self.input_lead_names[i]}')
            #axs[i].set_title(f"Channel {channel_ids[i]+1}")
            #if predicted:
            #    axs[i].plot(self.df.ecg_pred[idx][self.input_leads[i]], label="predicted")
        plt.savefig(f"figures/sample_plots/sample_{idx}_input_{tag}_annotation_{self.df.annotation[idx]}.png")

        #output leads
        fig, axs = plt.subplots(len(self.output_lead_idx),1)
        fig.suptitle(f'File {filename} output precordial leads') # or plt.suptitle('Main title')
        fig.tight_layout()
        for i in range(len(self.output_lead_idx)):
            axs[i].plot(ecg[self.output_lead_idx[i]], label="ground truth")
            axs[i].title.set_text(f'{self.output_lead_names[i]}')
            #axs[i].set_title(f"Channel {channel_ids[i]+1}")
            if ecg_pred.any():
                axs[i].plot(ecg_pred[self.output_lead_idx[i]], label="predicted")
        #figure legend
        axs[0].legend(loc="lower left", bbox_to_anchor=(0.75, 1.0))
        #save plot
        plt.savefig(f"figures/sample_plots/sample_{idx}_{tag}_annotation_{self.df.annotation[idx]}.png")


    def create_data_set(self, balance_data=False, centre_sample=False):
        filenames = glob.glob(self.path + '**/*.dat', recursive=True) #find all the filenames with .dat file extension

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['ecg_original'] = []
        set_dict['stft'] = []
        set_dict['stft_pred'] = []
        set_dict['ecg_pred'] = []
        set_dict['annotation'] = []
        set_dict['recording'] = []

        for file in tqdm(filenames, desc="processing beats"):
            tqdm.write(f"loading file: {file}")
            strips = self.load_file(file.removesuffix(".dat"), centre_sample)
            for ecg_strip, annotation_strip, ecg_strip_transformed, ecg_strip_original in zip(*strips):
                set_dict['ecg'].append(ecg_strip)
                set_dict['ecg_original'].append(ecg_strip_original)
                set_dict['stft'].append(ecg_strip_transformed)
                set_dict['stft_pred'].append([])
                set_dict['ecg_pred'].append([])
                set_dict['annotation'].append(annotation_strip)
                set_dict['recording'].append(file.split("/")[-1].removesuffix(".dat"))
        df = pd.DataFrame(set_dict)

        #balance the dataset if training
        #find random set of indices to drop from each beat type ("N", "V", "R") such that 
        #there is 2000 left (number of "A" beats)
        #drop all other kinds of beats
        if balance_data == True:

            print("Balancing dataset...")
            #collect random set of indices to 
            n_indices = df[df["annotation"] == "N"].index
            n_indices = shuffle(n_indices, random_state=0)[:len(n_indices)-2000]
            v_indices = df[df["annotation"] == "V"].index
            v_indices = shuffle(v_indices, random_state=1)[:len(v_indices)-2000]
            r_indices = df[df["annotation"] == "R"].index
            r_indices = shuffle(r_indices, random_state=2)[:len(r_indices)-2000]
            #drop indices
            indices = n_indices.to_list() + v_indices.to_list() + r_indices.to_list()
            df = df.drop(indices)
            #drop remaining beat types
            df = df[~df["annotation"].isin(["F", "j", "n", "Q", "S", "B"])]

        return df

    def reconstruct_signal(self, idx, prediction=False):
        """Reconstruct a signal from the FFT of the signal"""
        if prediction:
            stft = self.df.stft_pred[idx].squeeze()
        else:
            stft = self.df.stft[idx]
        ecg = self.df.ecg[idx]
        reconstructed_signal = []
        for i in range(0,len(stft),2): #increment i in steps of 2
            reconstructed_signal.append(self.SFT.istft(stft[i] + stft[i+1]*1j, k1=ecg.shape[1]))

        return np.array(reconstructed_signal)


    def predict(self, idx, model, device, default_dtype):
        """Make a prediction using a trained model
        Input: 
            idx (int): sample ID
            model (torch): model """

        input = torch.tensor(self.__getitem__(idx)['x']).to(device, dtype=default_dtype)
        prediction = model(input[None,:])
        self.df.loc[idx, "stft_pred"] = prediction.cpu().detach().numpy()
        reconstructed_signal = self.reconstruct_signal(idx, prediction=True)
        self.df.loc[idx, "ecg_pred"] = reconstructed_signal
        
        #self.plot(idx, predicted=True)

    def predict_all(self, model, device, default_dtype):
        print("Making predictions on entire set...")
        for i in tqdm(range(self.__len__())):
            self.predict(i, model, device, default_dtype)

    def matrix_corr(self, ecg, ecg_pred):
        #Flatten matrices to 1D
        ecg_flat = ecg.ravel()
        ecg_pred_flat = ecg_pred.ravel()
        #compute Pearson correlation
        corr, _ = pearsonr(ecg_flat, ecg_pred_flat)
        return corr

    def calculate_error(self):
        #only calculate the error on the unknown precordial limbs
        self.df["mse"] = self.df.apply(lambda row: mean_squared_error(row['ecg'][self.output_lead_idx], row['ecg_pred'][self.output_lead_idx]), axis=1)
        self.df["r2"] = self.df.apply(lambda row: r2_score(row['ecg'][self.output_lead_idx], row['ecg_pred'][self.output_lead_idx]), axis=1)
        self.df["pc"] = self.df.apply(lambda row: self.matrix_corr(row['ecg'][self.output_lead_idx], row['ecg_pred'][self.output_lead_idx]), axis=1)
        #calculate the error per channel
        self.df["mse_channel"] = self.df.apply(
            lambda row: [
                mean_squared_error(row['ecg'][i], row['ecg_pred'][i])
                for i in self.output_lead_idx
            ],
            axis=1
        )
        self.df["pc_channel"] = self.df.apply(
            lambda row: [
                self.matrix_corr(row['ecg'][i], row['ecg_pred'][i])
                for i in self.output_lead_idx
            ], 
            axis=1
        )

    def normalise_ecg_window(self, window):
        """
        window: ndarray of shape (12, n_samples) or (n_samples, 12)
        returns:
            window_norm
            fitted scaler
        """

        scaler = preprocessing.MinMaxScaler()
        window_norm = scaler.fit_transform(window.T).T

        return scaler

    def inverse_ecg_window(self, ecg, scaler):
        ecg_original_scale = scaler.inverse_transform(ecg.T).T
        return ecg_original_scale


    def reverse_normalise(self):
        #reverse the normalisation transfer in the preprocessing
        #"ecg_norm" is unnecessary and will never be used. Find a way to get rid if poss
        self.df["scaler"] = self.df.apply(
            lambda row: self.normalise_ecg_window(row['ecg_original']),
            axis=1
        )

        self.df['ecg_original_scale'] = self.df.apply(
            lambda row: self.inverse_ecg_window(row['ecg'], row['scaler']),
            axis=1
        )
        self.df['ecg_pred_original_scale'] = self.df.apply(
            lambda row: self.inverse_ecg_window(row['ecg'], row['scaler']),
            axis=1
        )






