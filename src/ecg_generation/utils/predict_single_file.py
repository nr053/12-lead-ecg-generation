"""This script can be used to make predictions on a single recording

- load the relevant recording into a dataset
- load the relevant pretrained model
- make predictions on the entire set
- save the MSE of each sample in a dataframe

"""

from ecg_generation.utils.unet import UNet
from ecg_generation.loading.incart_dataset import INCART_recording
from ecg_generation.utils.training_utils import Train_class, Test_class
import yaml
import random
import torch
from torch.utils.data import random_split
import os
import glob
import argparse

#arguments
parser = argparse.ArgumentParser(
                    prog='predict',
                    description='generate 12-lead ECG on a sample file using a specific model',
                    epilog='Text at the bottom of help')
parser.add_argument('recording')           # positional argument
parser.add_argument('model')           # positional argument
args = parser.parse_args()

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#set fixed seed
generator = torch.Generator().manual_seed(42)

#find the chosen recording 
matches = glob.glob(config['path_to_data'] + f"**/{args.recording}.dat", recursive=True)
filename = next(iter(matches), None)

#load dataset into custom data class
dataset = INCART_recording(filename, 
                        config
                        )

#recreate train/val/test split
set_size = len(dataset)
train_size = int(set_size*0.7) #70% of data
val_size = int(set_size*0.2) #20% of data
test_size = set_size - (train_size + val_size) #remaining 10% of data
subset_train, subset_val, subset_test = random_split(dataset, [train_size, val_size, test_size], generator=generator)

#GPU init
if torch.cuda.is_available(): #for NVIDIA graphics card
    device = torch.device("cuda")
    default_dtype = torch.float64
elif torch.backends.mps.is_available(): #M1 graphics card
    device = torch.device("mps")
    default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

#find model weights file
pattern = os.path.join(config['path_to_models'], '**', f'*{args.model}')
matches = glob.glob(pattern, recursive=True)
model_weights = next(iter(matches), None)

#load model
model = UNet().to(device)
model.load_state_dict(torch.load(model_weights, weights_only=True))
model.eval()

#predict and performance evaluation
dataset.predict_all(model, device, default_dtype)
dataset.calculate_error()

df_test = dataset.df.loc[subset_test.indices]

print(df_test.sort_values("mse").head())
print(df_test.sort_values("mse", ascending=False).head())

