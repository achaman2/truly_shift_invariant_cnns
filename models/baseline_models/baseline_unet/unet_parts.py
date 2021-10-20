""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.circular_pad_layer import circular_pad


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding_mode = 'zeros'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if padding_mode == 'circular':
            
            self.double_conv = nn.Sequential(
                circular_pad((1, 1, 1, 1)),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                
                circular_pad((1, 1, 1, 1)),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, padding_mode = 'zeros'):
        super().__init__()
        
        if padding_mode == 'circular':
            
            self.maxpool_conv = nn.Sequential(
                circular_pad((0, 1, 0, 1)),
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, padding_mode = padding_mode)
            )
            
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, padding_mode = padding_mode)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, padding_mode = 'zeros'):
        super().__init__()
        
        self.padding_mode = padding_mode

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, padding_mode = padding_mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, padding_mode = padding_mode)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



# instance norm based double conv codes


class DoubleConv_InstanceNorm(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding_mode = 'zeros'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if padding_mode == 'circular':
            
            self.double_conv = nn.Sequential(
                circular_pad((1, 1, 1, 1)),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                
                circular_pad((1, 1, 1, 1)),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down_InstanceNorm(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, padding_mode = 'zeros'):
        super().__init__()
        
        if padding_mode == 'circular':
            
            self.maxpool_conv = nn.Sequential(
                circular_pad((0, 1, 0, 1)),
                nn.MaxPool2d(2),
                DoubleConv_InstanceNorm(in_channels, out_channels, padding_mode = padding_mode)
            )
            
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv_InstanceNorm(in_channels, out_channels, padding_mode = padding_mode)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_InstanceNorm(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, padding_mode = 'zeros'):
        super().__init__()
        
        self.padding_mode = padding_mode

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_InstanceNorm(in_channels, out_channels, in_channels // 2, padding_mode = padding_mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_InstanceNorm(in_channels, out_channels, padding_mode = padding_mode)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)













