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
import time
from torch.utils.data import Subset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import scipy
from sklearn.utils import shuffle

class INCART_Dataset(data.Dataset):
    def __init__(self, path, window_size, hop, fs, channel_map):
        self.fs = fs
        self.channel_map = channel_map
        #fourier transform stuff
        w = scipy.signal.windows.hann(window_size)
        self.SFT = scipy.signal.ShortTimeFFT(w, hop=hop, fs=self.fs)
        #dataframe
        self.path = path
        self.df = self.create_data_set()

        #preprocessing
        self.scaler = preprocessing.MinMaxScaler()

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

        scaler = preprocessing.MinMaxScaler()

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
            
            #each input snippet is centred around a beat
            for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
                #there must be 0.5 seconds of signal either side of the beat position
                if (beat_position > 128) and (beat_position < signal_filtered.shape[0] - 128): 
                    ecg_strip = signal_filtered[beat_position-128:beat_position+128,:]
                    ecg_strip = np.swapaxes(ecg_strip, 0, 1)

                    ####
                    ####TESTING VARIOUS FORMS OF NORMALISATION 
                    ####

                    # #1. no normalisation of ECG strip
                    # ecg_strip_normalised = ecg_strip
                    
                    # #2. normalise ECG strip to unit vector
                    # ecg_strip_normalised = preprocessing.normalize(ecg_strip, axis=1) #normalise

                    #3. normalise ECG strip to [0,1] range
                    ecg_normalised_channels = []
                    for channel in ecg_strip: 
                        ecg_normalised_channels.append(scaler.fit_transform(channel.reshape(-1,1)))
                    ecg_strip_normalised = np.array(ecg_normalised_channels)[:,:,-1]
                    
                    ### END OF NORMALISATION ####

                    #append normalised strip
                    ecg_strips.append(ecg_strip_normalised)
                    
                    #append beat annotation
                    annotation_strips.append(beat_annotation)

                    #preprocessing step (fourier transform)
                    ecg_strip_transformed = self.preprocess(ecg_strip_normalised)
                    ecg_strips_transformed.append(ecg_strip_transformed)

            # #input snippets are chopped regardless of beat position
            # index = 0
            # while index <= signal_filtered.shape[0] - 256: #only include strips of 256 length
            #     ecg_strip = signal_filtered[index:index+256,:]
            #     ecg_strip = np.swapaxes(ecg_strip, 0, 1)

            #     ecg_normalised_channels = []
            #     for channel in ecg_strip: 
            #         ecg_normalised_channels.append(scaler.fit_transform(channel.reshape(-1,1)))
            #     ecg_strip_normalised = np.array(ecg_normalised_channels)[:,:,-1]

            #     #append normalised strip
            #     ecg_strips.append(ecg_strip_normalised)
                
            #     #append beat annotation
            #     beat_annotations = ""
            #     for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
            #         if index <= beat_position <= index+256:
            #             beat_annotations += beat_annotation    
            #     annotation_strips.append(beat_annotations)

            #     #preprocessing step (fourier transform)
            #     ecg_strip_transformed = self.preprocess(ecg_strip_normalised)
            #     ecg_strips_transformed.append(ecg_strip_transformed)

            #     index += 256

            


        return ecg_strips, annotation_strips, ecg_strips_transformed


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



    def plot(self, idx, channel_ids=[0,1,2,3,4,5,6,7,8,9,10,11], reconstructed=False, predicted=False):
        """Plot the ECG channels
        if reconstructed = True, the signal is reconstructed
        from the Fourier Transform"""
        if reconstructed:
            ecg = self.reconstruct_signal(idx)
        else: 
            ecg = self.df.ecg[idx]
        fig, axs = plt.subplots(len(channel_ids),1)
        fig.tight_layout()
        fig.suptitle(f"Reconstructed: {reconstructed}, beat type: {self.df.annotation[idx]}")
        for i in range(len(channel_ids)):
            axs[i].plot(ecg[channel_ids[i]], label="ground truth")
            #axs[i].set_title(f"Channel {channel_ids[i]+1}")
            if predicted:
                axs[i].plot(self.df.ecg_pred[idx][channel_ids[i]], label="predicted")
        
        
        #fig.legend()

        plt.savefig(f"figures/sample_plots/sample_{idx}_reconstructed_{reconstructed}_predicted_{predicted}.png")


    def create_data_set(self):
        #filenames = glob.glob(self.path + '**/*.dat', recursive=True) #find all the filenames with .dat file extension
        filenames = [self.path]

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['stft'] = []
        set_dict['stft_pred'] = []
        set_dict['ecg_pred'] = []
        set_dict['annotation'] = []
        set_dict['recording'] = []

        for file in tqdm(filenames, desc="Processing"):
            tqdm.write(f"loading file: {file}")
            ecg_strips, annotation_strips, ecg_strips_transformed = self.load_file(file.removesuffix(".dat"))
            for ecg_strip, annotation_strip, ecg_strip_transformed in zip(ecg_strips, annotation_strips, ecg_strips_transformed):
                set_dict['ecg'].append(ecg_strip)
                set_dict['stft'].append(ecg_strip_transformed)
                set_dict['stft_pred'].append([])
                set_dict['ecg_pred'].append([])
                set_dict['annotation'].append(annotation_strip)
                set_dict['recording'].append(file.split("/")[-1].removesuffix(".dat"))
        df = pd.DataFrame(set_dict)

        #balance the dataset
        #find indices to drop
        n_indices = df[df["annotation"] == "N"].index
        n_indices = shuffle(n_indices)[len(n_indices)-2000:]
        v_indices = df[df["annotation"] == "V"].index
        v_indices = shuffle(v_indices)[len(v_indices)-2000:]
        r_indices = df[df["annotation"] == "R"].index
        r_indices = shuffle(r_indices)[len(r_indices)-2000:]
        #drop indices
        indices = n_indices.to_list() + v_indices.to_list() + r_indices.to_list()
        df = df.drop(indices)


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

    def calculate_error(self):
        self.df["mse"] = self.df.apply(lambda row: mean_squared_error(row['ecg_pred'], row['ecg']), axis=1)



class INCART_Subset(Subset):
    def __init__(self, path, window_size, hop, fs, channel_map):
        super().__init__()
        self.df = self.df(dataset, indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


# class PTB_LOADER(INCART_LOADER):
#     def __init__(self, path, window_size, hop, fs, channel_map, patient):
#         self.fs = fs
#         self.channel_map = channel_map
#         #fourier transform stuff
#         w = scipy.signal.windows.hann(window_size)
#         self.SFT = scipy.signal.ShortTimeFFT(w, hop=hop, fs=self.fs)
#         #dataframe
#         self.df = self.create_data_set(path + "/PTB/physionet.org/files/ptbdb/1.0.0/" + patient)


#     def load_file(self, filename):
#         """The procedure to handle a record:
#             - create 0.5s strip with beat in the middle
#             - along with beat annotation
        
#         input: path to file (string)
#         ouput: array with signal data and accompanying annotation label
#                 possibly with metadata of patient / record ????

#         """
#         #load the file into a WFDB record object
#         signal = wfdb.rdrecord(filename)
#         annotation = wfdb.rdann(filename, extension="atr") 

#         #filter the signal
#         #b, a = scipy.signal.butter(5, [3, 50], btype='bandpass', fs=257)
#         #signal_filtered = scipy.signal.lfilter(b, a, signal.p_signal)
#         signal_filtered = scipy.signal.decimate(signal.p_signal, 2) #currently no filtering, now including downsampling for PTB set
        

#         #init lists to build set dict
#         ecg_strips = []
#         annotation_strips = []
#         ecg_strips_transformed = []

#         #check that the signal sampling freq matches the annotation s_freq and is 257
#         if signal.fs != annotation.fs:
#             print("reported sampling frequencies do not match")
#             return
#         elif signal.fs != self.fs:
#             print("sampling frequency is not 257")
#             return 
#         else:
            
#             for beat_position, beat_annotation in zip(annotation.sample, annotation.symbol):
#                 #there must be 0.5 seconds of signal either side of the beat position
#                 if (beat_position > 128) and (beat_position < signal_filtered.shape[0] - 128): 
#                     ecg_strip = signal_filtered[beat_position-128:beat_position+128,:]
#                     ecg_strip = np.swapaxes(ecg_strip, 0, 1)
#                     ecg_strip = preprocessing.normalize(ecg_strip, axis=1) #normalise
#                     ecg_strips.append(ecg_strip)
#                     annotation_strips.append(beat_annotation)

#                     #preprocessing step
#                     ecg_strip_transformed = self.preprocess(ecg_strip)
#                     ecg_strips_transformed.append(ecg_strip_transformed)

#         return ecg_strips, annotation_strips, ecg_strips_transformed, signal, annotation
