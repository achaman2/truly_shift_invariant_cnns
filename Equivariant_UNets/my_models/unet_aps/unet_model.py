""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .unet_aps_parts import *


class UNet_3down_aps(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels_list = [64, 128, 256, 512], filter_size = 3, bilinear=False, padding_mode = 'zeros'):
#         padding mode could be zeros or circular
        super(UNet_3down_aps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.padding_mode = padding_mode
        self.filter_size = filter_size

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(in_channels, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down2 = Down(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down3 = Down(inner_channels_list[2], inner_channels_list[3] // factor, padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)

        self.up1 = Up(inner_channels_list[3], inner_channels_list[2] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up2 = Up(inner_channels_list[2], inner_channels_list[1] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up3 = Up(inner_channels_list[1], inner_channels_list[0], bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.outc = OutConv(inner_channels_list[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2, poly2 = self.down1(x1)
        x3, poly3 = self.down2(x2)
        x4, poly4 = self.down3(x3)

        x = self.up1(x4, x3, poly4)
        x = self.up2(x, x2, poly3)
        x = self.up3(x, x1, poly2)
        out = self.outc(x)
        return out
    
    
class UNet_4down_aps(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels_list = [64, 128, 256, 512, 1024], filter_size = 3, bilinear=False, padding_mode = 'zeros', aps_criterion = 'l2'):
#         padding mode could be zeros or circular
        super(UNet_4down_aps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.padding_mode = padding_mode
        self.filter_size = filter_size
        
        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(in_channels, inner_channels_list[0], padding_mode = padding_mode)
        self.down1 = Down(inner_channels_list[0], inner_channels_list[1], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down2 = Down(inner_channels_list[1], inner_channels_list[2], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down3 = Down(inner_channels_list[2], inner_channels_list[3], padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)
        self.down4 = Down(inner_channels_list[3], inner_channels_list[4] // factor, padding_mode = padding_mode, filter_size = filter_size, aps_criterion = aps_criterion)

        self.up1 = Up(inner_channels_list[4], inner_channels_list[3] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up2 = Up(inner_channels_list[3], inner_channels_list[2] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up3 = Up(inner_channels_list[2], inner_channels_list[1] // factor, bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.up4 = Up(inner_channels_list[1], inner_channels_list[0], bilinear, padding_mode = padding_mode, filter_size = filter_size)
        self.outc = OutConv(inner_channels_list[0], out_channels)

    def forward(self, x):
        
        x1 = self.inc(x)
        x2, poly2 = self.down1(x1)
        x3, poly3 = self.down2(x2)
        x4, poly4 = self.down3(x3)
        x5, poly5 = self.down4(x4)

        x = self.up1(x5, x4, poly5)
        x = self.up2(x, x3, poly4)
        x = self.up3(x, x2, poly3)
        x = self.up4(x, x1, poly2)
        out = self.outc(x)
        
        return out
    
    
    
    