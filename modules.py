import torch.nn as nn
import torch

class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleConv, self).__init__()
        self.twoConvs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())
    
    def forward(self, x):
        x = self.twoConvs(x)
        return x

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.twoConvs = doubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feature_map = self.twoConvs(x)
        x = self.maxpool(feature_map)

        return x, feature_map

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.twoConvs = doubleConv(in_channels, out_channels)
        self.upConv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

    def forward(self, x, feature_map):
        x = self.upConv(x)
        x = torch.cat([x, feature_map], 1)
        x = self.twoConvs(x)

        return x
