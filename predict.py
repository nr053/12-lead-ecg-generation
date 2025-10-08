from unet import UNet
from load_incart import INCART_Dataset
import yaml
import random
import torch
from torch.utils.data import random_split
import os

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#set fixed seed
generator = torch.Generator().manual_seed(42)

#load dataset into custom data class
dataset = INCART_Dataset(config['path_to_data'] + "/INCART/files/incartdb/1.0.0/train/I59.dat", config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

#make train/test split
train_size = int(len(dataset)*0.5)
val_size = len(dataset) - train_size
#dataset_train = IN_subset(dataset, list(range(0,train_size)))
#dataset_test = IN_subset(dataset, list(range(train_size, len(dataset))))

subset_train, subset_val = random_split(dataset, [train_size, val_size], generator=generator)

#GPU init
if torch.cuda.is_available(): #for NVIDIA graphics card
    device = torch.device("cuda")
    default_dtype = torch.float64
elif torch.backends.mps.is_available(): #M1 graphics card
    device = torch.device("mps")
    default_dtype = torch.float32
torch.set_default_dtype(default_dtype)


#load model
model = UNet().to(device)
model.load_state_dict(torch.load("state_dicts/normalise/u_net_I59_Adam_lr_0.0001_epoch_196.pt", weights_only=True))
model.eval()

#predict and performance evaluation
dataset.predict_all(model, device, default_dtype)
dataset.calculate_error()

df_test = dataset.df.loc[subset_val.indices]

print(df_test.sort_values("mse").head())
print(df_test.sort_values("mse", ascending=False).head())