import torch
import torch.nn as nn
from torch.nn.functional import relu

class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(6, 50, kernel_size=3, padding=1) # output: 570x570x64
        self.conv2 = nn.Conv2d(50, 80, kernel_size=3, padding=1) # output: 568x568x64
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.fc1 = nn.Linear(960,256)
        self.fc2 = nn.Linear(256,32)

        #self.relu = nn.ReLU()  

    def forward(self, x):
        x1 = self.pool(nn.ReLU(self.conv1(x)))  # [B, 16, 8, 45] chatgpt # [128, 50, 16, 91]
        x2 = self.pool(nn.ReLU(self.conv2(x1)))  # [B, 32, 4, 22]
        x3 = torch.flatten(x2, start_dim=1)  # [B, 2816]
        x4 = nn.ReLU(self.fc1(x3))  # [B, 128]
        x5 = self.fc2(x4)  # [B, 32]
        return x5

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, 960)

        self.deconv1 = nn.ConvTranspose2d(80, 50, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(50, 24, kernel_size=3, padding=1)

        def forward(self, x):
            x1 = nn.ReLU(self.fc1(x))
            x2 = nn.ReLU(self.fc2(x1))
            x3 = x2.view(-1, 80, 4, 3)
            x4 = nn.ReLU(self.deconv1(x3))
            x4 = nn.ReLU(self.deconv2(x4))

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return out

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 16x16x3
        self.e1 = nn.Conv2d(6, 50, kernel_size=3, padding=1) # output: 570x570x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64
        self.activation_func = nn.ReLU()
        # input: 7x7x50
        self.e2 = nn.Conv2d(50, 80, kernel_size=3, padding=1) # output: 568x568x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64
        #input: 5x5x40
        self.flatten1 = nn.Flatten()
        #input: 1x1280
        self.fc1 = nn.Linear(960,256)
        #input: 1x256
        self.fc2 = nn.Linear(256,32)

        #DECODER
        #input: 1x32
        self.fc3 = nn.Linear(32, 256)
        #input: 1x256
        self.fc4 = nn.Linear(256, 960)
        #input: 1x1280
        self.reshape = nn.Unflatten(1, (80,4,3))
        #input 5x5x80
        self.d1 = nn.ConvTranspose2d(80, 50, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(size=(8,7))
        #input 7x7x50
        self.d2 = nn.ConvTranspose2d(50, 24, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(size=(16,14))

        self.fc5 = nn.Linear(5376, 5376)
        self.reshape_final = nn.Unflatten(1, (24,16,14))


    def forward(self, x):
        #first block of encoder
        x = self.e1(x)
        x = self.pool1(x)
        x = self.activation_func(x)
        #second block
        x = self.e2(x)
        x = self.pool2(x)
        x = self.activation_func(x)
        #flatten
        x = self.flatten1(x)
        #fully connected layers
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.fc2(x)
        #x = self.activation_func(x)

        #DECODER
        x = self.fc3(x)
        x = self.activation_func(x)
        x = self.fc4(x)
        x = self.activation_func(x)
        #unflatten
        x = self.reshape(x)
        #inverse conv
        x = self.d1(x)
        x = self.activation_func(x)
        #upsample
        #x = x.unsqueeze(1)
        x = self.upsample1(x)
        x = x.squeeze(1)  
        x = self.d2(x)
        x = self.activation_func(x)
        #print(f"size after second decoder: {x.shape}")
        #x = x.unsqueeze(1)
        #print(f"size after squeeze: {x.shape}")
        x = self.upsample2(x)
        #print(f"size after upsample: {x.shape}")
        #x = x.squeeze(1)
        #print(f"size after second upsample: {x.shape}")
        x = self.flatten1(x)
        #print(f"size after flatten: {x.shape}")
        x = self.fc5(x)
        x = self.reshape_final(x)


        return x
        


        
