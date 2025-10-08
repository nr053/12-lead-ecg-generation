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
import yaml


class Train_class():
    def __init__(self, model_tag, train_set, val_set):
        #load config
        config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))        

        #load dataset into custom data class
        print("")
        print("---Loading train set---")
        print("")
        self.set_train = train_set
        print("")
        print("---Loading validation set---")
        print("")
        self.set_val = val_set
        

        #dataloaders
        self.dataloader_train = DataLoader(self.set_train, batch_size=config['batch_size'], shuffle=True)
        self.dataloader_val = DataLoader(self.set_val, batch_size=config['batch_size'], shuffle=True)

        #GPU stuff
        if torch.cuda.is_available(): #for NVIDIA graphics card
            self.device = torch.device("cuda")
            self.default_dtype = torch.float64
        elif torch.backends.mps.is_available(): #M1 graphics card
            self.device = torch.device("mps")
            self.default_dtype = torch.float32
        torch.set_default_dtype(self.default_dtype)
        print(f"Using device {self.device}")

        #model definition and hyperparameters
        self.model = UNet().to(self.device)
        self.learning_rate = config["learning_rate"]
        #drop out = 0.5, 0.1
        self.loss_fn = nn.MSELoss() #or RMSE???
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_tag = model_tag

    def train_loop(self, n_epochs):
        #init empty lists to track loss
        epoch_loss_train = []
        epoch_loss_val = []
        current_epoch = 1

        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            #train step
            self.model.train()
            batch_loss_train = []
            #loop = tqdm(dataloader)
            for sample in self.dataloader_train:
                #make prediction
                prediction = self.model(sample['x'].to(self.device, dtype=self.default_dtype)) #first 3 leads as input
                
                #calculate loss
                loss = self.loss_fn(prediction.to(self.device, dtype=self.default_dtype), sample['y'].to(self.device, dtype=self.default_dtype)) #all 12 leads as ground truth
                batch_loss_train.append(loss.item())

                #backpropogation step
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            epoch_loss_train.append(mean(batch_loss_train))

            #validation step
            self.model.eval()
            batch_loss_val = []
            with torch.no_grad():
                #loop = tqdm(dataloader_val)
                for sample in self.dataloader_val:
                    #make prediction
                    prediction = self.model(sample['x'].to(self.device, dtype=self.default_dtype))

                    #calculate loss
                    loss = self.loss_fn(prediction.to(self.device, dtype=self.default_dtype), sample['y'].to(self.device, dtype=self.default_dtype)) #all 12 leads as ground truth
                    batch_loss_val.append(loss.item())

            #track mean test batch loss over epochs    
            current_loss = mean(batch_loss_val)    

            if current_epoch == 1: #on first epoch add epoch loss to empty list and set best epoch == 1
                best_epoch = current_epoch
                best_model_state = self.model.state_dict()
            elif current_loss < min(epoch_loss_val): #check if current epoch yielded better performance
                best_epoch = current_epoch
                best_model_state = self.model.state_dict()
            current_epoch += 1
            epoch_loss_val.append(current_loss)

        #save the best model state dict
        model_name = f"state_dicts/normalise/u_net_{self.model_tag}_{self.optimiser.__class__.__name__}_lr_{self.learning_rate}_epoch_{best_epoch}.pt"
        torch.save(best_model_state, model_name)
        self.best_model_state = best_model_state

        #plot loss
        plt.figure()
        plt.plot(epoch_loss_train, label="train")
        plt.plot(epoch_loss_val, label="val")
        plt.legend()
        plt.ylabel(f"Loss: {self.loss_fn}")
        plt.title(f"Model: {self.model_tag}, LR: {self.learning_rate}, optimiser: {self.optimiser.__class__.__name__}, best epoch: {best_epoch}")
        plt.savefig(f"figures/loss_plots/normalise/loss_plot_{self.model_tag}_LR_{self.learning_rate}.png")

        return model_name

class Test_class():
    def __init__(self, model_tag, dataset):
        #load config
        config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))   

        #load dataset
        self.dataset = dataset
        
        #dataloaders
        self.dataloader_test = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True)

        #GPU stuff
        if torch.cuda.is_available(): #for NVIDIA graphics card
            self.device = torch.device("cuda")
            self.default_dtype = torch.float64
        elif torch.backends.mps.is_available(): #M1 graphics card
            self.device = torch.device("mps")
            self.default_dtype = torch.float32
        torch.set_default_dtype(self.default_dtype)
        print(f"Using device {self.device}")
        
        self.model_tag = model_tag
        
        #model definition and hyperparameters
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_tag, weights_only=True))
        self.model.eval()

    def predict(self):
        self.dataset.predict_all(self.model, self.device, self.default_dtype)
        self.dataset.calculate_error()