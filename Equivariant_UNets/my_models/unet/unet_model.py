""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet_3down(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels_list = [64, 128, 256, 512], bilinear=False, padding_mode = 'zeros'):
#         padding mode could be zeros or circular
        super(UNet_3down, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.padding_mode = padding_mode

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode)
        self.down2 = Down(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode)
        self.down3 = Down(inner_channels_list[2], inner_channels_list[3]// factor, padding_mode = padding_mode)
        
        self.up1 = Up(inner_channels_list[3], inner_channels_list[2] // factor, bilinear, padding_mode = padding_mode)
        self.up2 = Up(inner_channels_list[2], inner_channels_list[1] // factor, bilinear, padding_mode = padding_mode)
        self.up3 = Up(inner_channels_list[1], inner_channels_list[0], bilinear, padding_mode = padding_mode)
        self.outc = OutConv(inner_channels_list[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.outc(x)
        return out
    
    
class UNet_4down(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels_list = [64, 128, 256, 512, 1024], bilinear=False, padding_mode = 'zeros'):
        super(UNet_4down, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.padding_mode = padding_mode
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode)
        self.down2 = Down(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode)
        self.down3 = Down(inner_channels_list[2], inner_channels_list[3], padding_mode = padding_mode)
        self.down4 = Down(inner_channels_list[3], inner_channels_list[4] // factor, padding_mode = padding_mode) 
        
        self.up1 = Up(inner_channels_list[4], inner_channels_list[3] // factor, bilinear, padding_mode = padding_mode)
        self.up2 = Up(inner_channels_list[3], inner_channels_list[2] // factor, bilinear, padding_mode = padding_mode)
        self.up3 = Up(inner_channels_list[2], inner_channels_list[1] // factor, bilinear, padding_mode = padding_mode)
        self.up4 = Up(inner_channels_list[1], inner_channels_list[0], bilinear, padding_mode = padding_mode)
        self.outc = OutConv(inner_channels_list[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out



#U-Net with instance norm rather than batchnorm. 

class UNet_4down_InstanceNorm(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels_list = [64, 128, 256, 512, 1024], bilinear=False, padding_mode = 'zeros'):
        super(UNet_4down_InstanceNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.padding_mode = padding_mode
        factor = 2 if bilinear else 1

        self.inc = DoubleConv_InstanceNorm(in_channels, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down_InstanceNorm(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode)
        self.down2 = Down_InstanceNorm(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode)
        self.down3 = Down_InstanceNorm(inner_channels_list[2], inner_channels_list[3], padding_mode = padding_mode)
        self.down4 = Down_InstanceNorm(inner_channels_list[3], inner_channels_list[4] // factor, padding_mode = padding_mode) 
        
        self.up1 = Up_InstanceNorm(inner_channels_list[4], inner_channels_list[3] // factor, bilinear, padding_mode = padding_mode)
        self.up2 = Up_InstanceNorm(inner_channels_list[3], inner_channels_list[2] // factor, bilinear, padding_mode = padding_mode)
        self.up3 = Up_InstanceNorm(inner_channels_list[2], inner_channels_list[1] // factor, bilinear, padding_mode = padding_mode)
        self.up4 = Up_InstanceNorm(inner_channels_list[1], inner_channels_list[0], bilinear, padding_mode = padding_mode)
        self.outc = OutConv(inner_channels_list[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

