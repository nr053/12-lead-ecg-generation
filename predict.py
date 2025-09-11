from unet import UNet
from load_incart import INCART_LOADER, IN_subset
import yaml
import random
import torch
import os

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset = INCART_LOADER(config['path_to_data'], config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

#make train/test split
train_size = int(len(dataset)*0.9)
dataset_train = IN_subset(dataset, list(range(0,train_size)))
dataset_val = IN_subset(dataset, list(range(train_size, len(dataset))))

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
model.load_state_dict(torch.load("state_dicts/u_net_Adam_lr_0.0001_epoch_405_val.pt", weights_only=True))
model.eval()

#get collection of random samples
#dataset.predict(1501,model, device, default_dtype)

print(dataset_train.__len__())
dataset_train.predict_all(model, device, default_dtype)
dataset_train.calculate_error()

dataset_val.predict_all(model, device, default_dtype)
dataset_val.calculate_error()