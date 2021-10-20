
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.circular_pad_layer import circular_pad


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, sinc_mode = False, stride=2, pad_off=0, circular_flag = False):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.circular_flag = circular_flag    #use circular padding when this flag is on
        self.sinc_mode = sinc_mode
        
        if self.circular_flag == True:
            pad_type = 'circular'
        
        
        if not self.sinc_mode:
            
            if(self.filt_size==1):
                a = np.array([1.,])
            elif(self.filt_size==2):
                a = np.array([1., 1.])
            elif(self.filt_size==3):
                a = np.array([1., 2., 1.])
            elif(self.filt_size==4):    
                a = np.array([1., 3., 3., 1.])
            elif(self.filt_size==5):    
                a = np.array([1., 4., 6., 4., 1.])
            elif(self.filt_size==6):    
                a = np.array([1., 5., 10., 10., 5., 1.])
            elif(self.filt_size==7):    
                a = np.array([1., 6., 15., 20., 15., 6., 1.])

            filt = torch.Tensor(a[:,None]*a[None,:])
            filt = filt/torch.sum(filt)
            self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

            self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        
        if self.sinc_mode == True:
            
            sinc_out = sinc_low_pass(inp)
            return sinc_out[:, :, ::self.stride, ::self.stride]
        
        
        else:
            if(self.filt_size==1):
                if(self.pad_off==0):
                    return inp[:,:,::self.stride,::self.stride]    
                else:
                    return self.pad(inp)[:,:,::self.stride,::self.stride]
            else:
                return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

            
def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    elif(pad_type == 'circular'):
        PadLayer = circular_pad
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


def sinc_low_pass(x):
    
    B, C, N = x.shape[0:3]
    
#     zeros = torch.zeros(x.shape).float().cuda()
    
#     signal_cat = torch.stack((x, zeros), dim = 4)
    ff_x = torch.fft.fftn(x, dim = (2, 3))
    
    low = int(np.ceil(N/4))
    high = N - low
    
    ff_x[:, :, low : high+1, :] = 0.0
    ff_x[:, :, :, low : high+1] = 0.0
    
    low_pass_out = torch.real(torch.fft.ifftn(ff_x, dim = (2, 3))).to(x.dtype)
  
    
    return low_pass_out
    

class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer