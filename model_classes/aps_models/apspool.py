import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model_classes.models_for_cifar10.circular_pad_layer import circular_pad


class ApsPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, circular_flag = False, apspool_criterion = 'l2'):
        super(ApsPool, self).__init__()
        self.filt_size = filt_size
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.circular_flag = circular_flag    #use circular padding when this flag is on
        
        self.apspool_criterion = apspool_criterion
        
        if self.circular_flag == True:
            pad_type = 'circular'

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

        
    def forward(self, input_to_pool):
        
        if isinstance(input_to_pool, dict):
#         I had to construct this weird way of taking input to accomodate for nn sequential in resnet_aps
            inp, polyphase_indices = input_to_pool['inp'], input_to_pool['polyphase_indices']
    
        else:
#             this is the case when polyphase indices are not pre-defined
            inp = input_to_pool
            polyphase_indices = polyphase_comp_select(inp, apspool_criterion = self.apspool_criterion)

        if(self.filt_size==1):
            return aps_downsample(aps_pad(inp), self.stride, polyphase_indices)
            
        else:
            
            blurred_inp = F.conv2d(self.pad(inp), self.filt, stride = 1, groups=inp.shape[1])
            return aps_downsample(aps_pad(blurred_inp), self.stride, polyphase_indices)


def aps_downsample(x, stride, polyphase_indices):
    
    if stride ==1:
        return x
    
    elif stride ==2:
    
        xpoly_0 = x[:, :, ::stride, ::stride]
        xpoly_1 = x[:, :, 1::stride, ::stride]
        xpoly_2 = x[:, :, ::stride, 1::stride]
        xpoly_3 = x[:, :, 1::stride, 1::stride]

        xpoly_combined = torch.stack([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim = 4)

        B = xpoly_combined.shape[0]
        output = xpoly_combined[torch.arange(B), :, :, :, polyphase_indices]
        
        return output
        
    elif stride>2:
        raise ValueError('ApsPool implementation currently does not support stride>2')
    
        

def polyphase_comp_select(x, apspool_criterion = 'l2'):
    
    stride = 2
    xpoly_0 = aps_pad(x)[:, :, ::stride, ::stride].reshape(x.shape[0], -1)
    
    xpoly_1 = aps_pad(x)[:, :, 1::stride, ::stride].reshape(x.shape[0], -1)
    
    xpoly_2 = aps_pad(x)[:, :, ::stride, 1::stride].reshape(x.shape[0], -1)
    
    xpoly_3 = aps_pad(x)[:, :, 1::stride, 1::stride].reshape(x.shape[0], -1)
    
    
    if apspool_criterion.endswith('min'):
        
        criterion = apspool_criterion[:-3]
        
        if criterion == 'l2':
            norm_ind = 2

        elif criterion == 'l1':
            norm_ind = 1


        else:
            raise ValueError('Unknown criterion choice')


        norm0 = torch.norm(xpoly_0, dim = 1, p = norm_ind)
        norm1 = torch.norm(xpoly_1, dim = 1, p = norm_ind)
        norm2 = torch.norm(xpoly_2, dim = 1, p = norm_ind)
        norm3 = torch.norm(xpoly_3, dim = 1, p = norm_ind)

        all_norms = torch.stack([norm0, norm1, norm2, norm3], dim = 1)

        return torch.argmin(all_norms, dim = 1)
        
    else:
    
        if apspool_criterion == 'l_infty':

            max_0 = torch.max(xpoly_0.abs(), dim = 1).values
            max_1 = torch.max(xpoly_1.abs(), dim = 1).values
            max_2 = torch.max(xpoly_2.abs(), dim = 1).values
            max_3 = torch.max(xpoly_3.abs(), dim = 1).values


            all_max = torch.stack([max_0, max_1, max_2, max_3], dim = 1)

            return torch.argmax(all_max, dim = 1)


        elif apspool_criterion == 'l2':
            norm_ind = 2

        elif apspool_criterion == 'l1':
            norm_ind = 1


        else:
            raise ValueError('Unknown criterion choice')


        norm0 = torch.norm(xpoly_0, dim = 1, p = norm_ind)
        norm1 = torch.norm(xpoly_1, dim = 1, p = norm_ind)
        norm2 = torch.norm(xpoly_2, dim = 1, p = norm_ind)
        norm3 = torch.norm(xpoly_3, dim = 1, p = norm_ind)

        all_norms = torch.stack([norm0, norm1, norm2, norm3], dim = 1)

        return torch.argmax(all_norms, dim = 1)

    

def aps_pad(x):
    
    N1, N2 = x.shape[2:4]
    
    if N1%2==0 and N2%2==0:
        return x
    
    if N1%2!=0:
        x = F.pad(x, (0, 0, 0, 1), mode = 'circular')
    
    if N2%2!=0:
        x = F.pad(x, (0, 1, 0, 0), mode = 'circular')
    
    return x
        

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

