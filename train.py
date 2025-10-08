"""
This script can be used to train a UNet model to reconstruct 12-lead ECG
from a subset of leads.

- Load data
- define model hyperparameters
- train model
- save loss plot
- save model (optional)
"""

from unet import UNet
from load_incart import INCART_LOADER
from training_utils import Train_class, Test_class

train = Train_class("new_model")
train.train_loop()

