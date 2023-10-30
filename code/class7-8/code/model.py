from torch import nn as nn
from torch.nn import functional as F


class MedNet(nn.Module):
    def __init__(self, xDim, yDim, numC):  # Pass image dimensions and number of labels when initializing a model
        super(MedNet, self).__init__()  # Extends the basic nn.Module to the MedNet class

        filter_size = 3

        self.cnv1 = nn.Conv2d(1, 8, filter_size,padding=1)
        self.act1=nn.ReLU()
        self.pooling1=nn.MaxPool2d(2)

        self.cnv2 = nn.Conv2d(8, 16, filter_size,padding=1)
        self.act2 = nn.ReLU()
        self.pooling2=nn.MaxPool2d(2)

        self.cnv3= nn.Conv2d(16, 32, filter_size,padding=1)
        self.act3 = nn.ReLU()
        self.pooling3=nn.MaxPool2d(2)

        self.cnv4= nn.Conv2d(32, 64, filter_size,padding=1)
        self.act4 = nn.ReLU()
        self.pooling4=nn.MaxPool2d(2)

        self.flat=nn.Flatten()

        self.ful1 = nn.Linear(4*4*64, 128)
        self.ful2 = nn.Linear(128, 64)
        self.ful3 = nn.Linear(64, numC)

    def forward(self, x):
        x=self.cnv1(x)
        x=self.act1(x)
        x=self.pooling1(x)
        x=self.cnv2(x)
        x=self.act2(x)
        x=self.pooling2(x)
        x=self.cnv3(x)
        x=self.act3(x)
        x=self.pooling3(x)
        x=self.cnv4(x)
        x=self.act4(x)
        x=self.pooling4(x)
        x=self.flat(x)
        x=self.ful1(x)
        x=self.ful2(x)
        x=self.ful3(x)
        return  x