import torch.nn as nn
import torch
import numpy as np 
from modules import up, down, doubleConv

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.down1 = down(in_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.middle = doubleConv(512, 1024)

        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x, feature_map1 = self.down1(x)
        x, feature_map2 = self.down2(x)
        x, feature_map3 = self.down3(x)
        x, feature_map4 = self.down4(x)

        x = self.middle(x)

        x = self.up1(x, feature_map4)
        x = self.up2(x, feature_map3)
        x = self.up3(x, feature_map2)
        x = self.up4(x, feature_map1)

        x = self.last(x)
  
        return x