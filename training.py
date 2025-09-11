"""
This script can be used to train a UNet model to reconstruct 12-lead ECG
from a subset of leads.

- Load data
- define model hyperparameters
- train model
- save loss plot
- save model (optional)

The training and testing loops can be written as functions for imports to use in later training
scripts but are for now written explicitly to facilitate interaction with variables in the terminal
for debugging purposes.
"""

from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm
import torch
from statistics import mean
import matplotlib.pyplot as plt
import os
import math
from unet import UNet

from load_incart import INCART_LOADER, IN_subset
import yaml

#load config
config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset = INCART_LOADER(config['path_to_data'], config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

#make train/test split
train_size = int(len(dataset)*0.9)
dataset_train = IN_subset(dataset, list(range(0,train_size)))
dataset_val = IN_subset(dataset, list(range(train_size, len(dataset))))

#dataloaders
#dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=config['batch_size'], shuffle=True)

#GPU stuff
if torch.cuda.is_available(): #for NVIDIA graphics card
    device = torch.device("cuda")
    default_dtype = torch.float64
elif torch.backends.mps.is_available(): #M1 graphics card
    device = torch.device("mps")
    default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
print(f"Using device {device}")
#model definition and hyperparameters
model = UNet().to(device)
learning_rate = 10e-5
n_epochs = 10
#drop out = 0.5, 0.1
loss_fn = nn.MSELoss() #or RMSE???
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#init empty lists to track loss
epoch_loss_train = []
epoch_loss_val = []
current_epoch = 1


for epoch in tqdm(range(n_epochs)):
    #train step
    model.train()
    batch_loss_train = []
    #loop = tqdm(dataloader)
    for sample in dataloader_train:
        #make prediction
        prediction = model(sample['x'].to(device, dtype=default_dtype)) #first 3 leads as input
        
        #calculate loss
        loss = loss_fn(prediction.to(device, dtype=default_dtype), sample['y'].to(device, dtype=default_dtype)) #all 12 leads as ground truth
        batch_loss_train.append(loss.item())

        #backpropogation step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    epoch_loss_train.append(mean(batch_loss_train))
    
    #validation step
    model.eval()
    batch_loss_val = []
    with torch.no_grad():
        #loop = tqdm(dataloader_val)
        for sample in dataloader_val:
            #make prediction
            prediction = model(sample['x'].to(device, dtype=default_dtype))

            #calculate loss
            loss = loss_fn(prediction.to(device, dtype=default_dtype), sample['y'].to(device, dtype=default_dtype)) #all 12 leads as ground truth
            batch_loss_val.append(loss.item())

    #track mean test batch loss over epochs    
    current_loss = mean(batch_loss_val)    

    if current_epoch == 1: #on first epoch add epoch loss to empty list and set best epoch == 1
        best_epoch = current_epoch
    elif current_loss < min(epoch_loss_val): #check if current epoch yielded better performance
        best_epoch = current_epoch
        best_model_state = model.state_dict()
    current_epoch += 1
    epoch_loss_val.append(current_loss)

#save the best model state dict
torch.save(best_model_state, f"state_dicts/u_net_{optimiser.__class__.__name__}_lr_{learning_rate}_epoch_{best_epoch}_val_norm.pt")

#plot loss
plt.plot(epoch_loss_train, label="train")
plt.plot(epoch_loss_val, label="val")
plt.legend()
plt.ylabel(f"Loss: {loss_fn}")
plt.title(f"LR: {learning_rate}, optimiser: {optimiser.__class__.__name__}, best epoch: {best_epoch}")
plt.savefig(f"figures/loss_plots/loss_plot_LR_{learning_rate}_val_norm.png")



model.load_state_dict(best_model_state)
model.eval()

#get collection of random samples
#dataset_val.predict(3,model, device, default_dtype)

dataset_val.predict_all(model, device, default_dtype)
dataset_val.calculate_error()