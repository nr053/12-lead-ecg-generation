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

        # input: 7x7x50
        self.e2 = nn.Conv2d(50, 80, kernel_size=3, padding=1) # output: 568x568x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        #input: 5x5x40
        self.flatten1 = nn.Flatten(0,2)

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
        self.reshape = nn.Unflatten(0, (80,4,3))

        #input 5x5x80
        self.d1 = nn.ConvTranspose2d(80, 50, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(size=(8,7))

        #input 7x7x50
        self.d2 = nn.ConvTranspose2d(50, 24, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(size=(16,14))
        # # input: 284x284x64
        # self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        # self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # # input: 140x140x128
        # self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        # self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # # input: 68x68x256
        # self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        # self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # # input: 32x32x512
        # self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        # self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # # Decoder
        # self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        # self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # # Output layer
        # self.outconv = nn.Conv2d(64, n_class, kernel_size=1)


    def forward(self, x):
        #first block of encoder
        print(f"size of input: {x.shape}")
        x = self.e1(x)
        print(f"size after first conv: {x.shape}")
        x = self.pool1(x)
        print(f"size after first pool block: {x.shape}")
        #second block
        x = self.e2(x)
        print(f"size after second conv: {x.shape}")
        x = self.pool2(x)
        print(f"size after second pool block: {x.shape}")
        #flatten
        x = self.flatten1(x)
        print(f"size after flatten: {x.shape}")
        #fully connected layers
        x = self.fc1(x)
        print(f"size after first fully connected: {x.shape}")
        x = self.fc2(x)
        print(f"size after second fully connected: {x.shape}")

        #DECODER
        x = self.fc3(x)
        print(f"size after third fully connected: {x.shape}")
        x = self.fc4(x)
        print(f"size after fourth fully connected: {x.shape}")
        #unflatten
        x = self.reshape(x)
        print(f"size after reshape: {x.shape}")
        #inverse conv
        x = self.d1(x)
        print(f"size after first decoder: {x.shape}")
        #upsample
        x = x.unsqueeze(1)
        print(f"size after squeeze: {x.shape}")
        x = self.upsample1(x)
        print(f"size after first upsample: {x.shape}")      
        x = x.squeeze(1)  
        x = self.d2(x)
        print(f"size after second decoder: {x.shape}")
        x = x.unsqueeze(1)
        x = self.upsample2(x)
        x = x.squeeze(1)
        print(f"size after second upsample: {x.shape}")


        return x
        


        
