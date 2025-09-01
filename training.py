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

from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch
from statistics import mean
import matplotlib.pyplot as plt
import os
import math
from unet import UNet

from load_incart import INCART_LOADER
import yaml

config = yaml.safe_load(open(f"{os.getcwd()}/config.yaml"))

#load dataset into custom data class
dataset = INCART_LOADER(config['path_to_data'], config["window_size"], config["hop"], config["s_freq"], config["channel_map"])

#make train/test split


#dataloaders
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
# train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
# test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
#dataloaders
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
# train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
# test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

#define model and hyperparameters

#GPU stuff
if torch.cuda.is_available(): #for NVIDIA graphics card
    device = torch.device("cuda")
    default_dtype = torch.float64
elif torch.backends.mps.is_available(): #M1 graphics card
    device = torch.device("mps")
    default_dtype = torch.float32
    torch.set_default_dtype(torch.float32)
#model definition and hyperparameters
model = UNet().to(device)
learning_rate = 10e-6
n_epochs = 10000
#drop out = 0.5, 0.1
loss_fn = nn.MSELoss() #or RMSE???
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)




# # # def train_loop(model, dataloader, loss_fn):
# # #     batch_loss = []
# # #     loop = tqdm(dataloader)
# # #     for sample in loop:
# # #         #make prediction
# # #         sample['prediction'] = model(sample['X'])
        
# # #         #calculate loss
# # #         loss = loss_fn(sample['prediction'], sample['y'])
# # #         batch_loss.append(loss)
# # #         #calculate accuracy


# # #         #backpropogation step
# # #         optimiser.zero_grad()
# # #         loss.backward()
# # #         optimiser.step()

# # #     return batch_loss

epoch_loss = []
#epoch_loss_test = []
epoch_number = 1


for epoch in tqdm(range(n_epochs)):
    #train step
    print("")
    print("-------- Training loop ----------")
    print("")
    model.train()
    
    batch_loss = []
    loop = tqdm(dataloader)
    

    for sample in loop:
        #make prediction
        prediction = model(sample['x'].to(device, dtype=default_dtype)) #first 3 leads as input
        
        #calculate loss
        loss = loss_fn(prediction.to(device, dtype=default_dtype), sample['y'].to(device, dtype=default_dtype)) #all 12 leads as ground truth
        batch_loss.append(loss.item())

        #backpropogation step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        #data.add_prediction(sample)

    #track mean train batch loss over epochs
    #train_loss = batch_loss_train
    #train_loss = train_loop(model, dataloader, loss_fn)
    epoch_loss.append(mean(batch_loss))


#     #test step
#     print("")
#     print("-------- Testing loop ----------")
#     print("")
#     model.eval()
#     batch_loss_test = []

#     with torch.no_grad():
#         loop = tqdm(test_dataloader)
#         for sample in loop:
#             #make prediction
#             sample['prediction'] = model(sample['X'].cuda())

#             #calculate loss
#             loss = loss_fn(sample['prediction'].cuda(), sample['y'][:,None,:].cuda())
#             batch_loss_test.append(loss.item())

#             #accuracy?

#             data.add_prediction(sample)

#     #track mean test batch loss over epochs        
#     epoch_loss_test.append(mean(batch_loss_test))
 
#     if epoch_number % 100 == 0:
#         plt.plot(epoch_loss_train)
#         plt.plot(epoch_loss_test)
#         plt.savefig("loss_plot" + str(epoch_number) + ".png")
#     epoch_number+=1


plt.plot(epoch_loss)
plt.savefig("loss_plot.png")




# model.eval()
# torch.no_grad()



    
#torch.save(model.state_dict(), "CNN.pt")




