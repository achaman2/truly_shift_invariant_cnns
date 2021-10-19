
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
from models.aps_models.apspool import ApsPool

from models.circular_pad_layer import circular_pad


# import apspool, circular_pad_layer
# from apspool import ApsPool, polyphase_comp_select
# from circular_pad_layer import circular_pad

__all__ = [ 'ResNet' , 'resnet18']


def conv3x3(in_planes, out_planes, stride=1, groups=1, conv_pad_type = 'circular'):
    """3x3 convolution with padding"""
    
    return nn.Sequential(circular_pad([1, 1, 1, 1]),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=0, groups=groups, bias=False)
                            )

def conv7x7(in_planes, out_planes, stride=1, groups=1, conv_pad_type = 'circular'):
    """3x3 convolution with padding"""
    
    return nn.Sequential(circular_pad([3, 3, 3, 3]),
                             nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                         padding=0, groups=groups, bias=False)
                            )

        

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
                
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, filter_size=1, N = None, conv_pad_type = 'circular', apspool_criterion = 'l2'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.N = N
        
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, conv_pad_type = conv_pad_type)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.circular_flag = False
        if conv_pad_type == 'circular':
            self.circular_flag = True
            
        self.apspool_criterion = apspool_criterion
        
        self.conv2 = conv3x3(planes, planes, conv_pad_type = conv_pad_type)
        
        if stride>1:
            
            self.aps_down = ApsPool(planes, filt_size = filter_size, stride = stride, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion, return_poly_indices = True, N = self.N)
            
        
#         if(stride==1):
#             self.conv2 = conv3x3(planes,planes, conv_pad_type = conv_pad_type)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.stride >1:
            out, indices = self.aps_down(out)
            
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.stride >1:
                identity = self.downsample({'inp': x, 'polyphase_indices': indices})
            else:
                identity = self.downsample(x)
                
        N_out = out.shape[2]
        N_id = identity.shape[2]
        
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, filter_size=1, conv_pad_type = 'zeros', apspool_criterion = 'l2', N = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups
        self.circular_flag = False
        if conv_pad_type == 'circular':
            self.circular_flag = True
            
        self.apspool_criterion = apspool_criterion
        self.N = N
#         if apspool_criterion !='l2':
            
            
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups=groups, conv_pad_type = conv_pad_type) # stride moved
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1(width, planes * self.expansion)
        
            
        if self.stride>1:
            self.aps_down = ApsPool(width, filt_size=filter_size, stride=stride, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion, return_poly_indices = True, N = self.N)
            
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.stride>1:
            out, indices = self.aps_down(out)
            
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if self.stride >1:
                identity = self.downsample({'inp': x, 'polyphase_indices': indices})
            else:
                identity = self.downsample(x)
                
        
        N_out = out.shape[2]
        N_id = identity.shape[2]
        
        
        if N_id!=N_out:
            pad_extra = N_out - N_id
            identity = F.pad(identity, (0, pad_extra, 0, pad_extra) , mode = 'replicate')
            
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, layer_channels = [64, 128, 256, 512], width_per_group=64, norm_layer=None, filter_size=1, 
                 pool_only=False, conv_pad_type = 'circular', apspool_criterion = 'l2', N_in = None):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = layer_channels[0]
        self.base_width = width_per_group
        self.conv_pad_type = conv_pad_type
        self.circular_flag = False
        if conv_pad_type == 'circular':
            self.circular_flag = True
        
        self.apspool_criterion = apspool_criterion
        
        
        if(pool_only):
            self.conv1 = conv7x7(3, self.inplanes, stride=2)
        else:
            self.conv1 = conv7x7(3, self.inplanes, stride=1)
            
        
        
        if(pool_only):
            self.maxpool = nn.Sequential(*[circular_pad([0, 1, 0 ,1]), nn.MaxPool2d(kernel_size=2, stride=1), 
                ApsPool(channels = self.inplanes, filt_size=filter_size, stride=2, apspool_criterion = self.apspool_criterion, return_poly_indices = False)]) 
                       
        else:
            self.maxpool = nn.Sequential(*[ApsPool(channels = self.inplanes, filt_size=filter_size, stride=2, apspool_criterion = self.apspool_criterion, return_poly_indices = False),circular_pad([0, 1, 0 ,1]),  
                                           nn.MaxPool2d(kernel_size=2, stride=1),
                                           ApsPool(channels = self.inplanes, filt_size=filter_size, stride=2, apspool_criterion = self.apspool_criterion, return_poly_indices = False)])
                
                
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        if N_in is not None:
            N_layers = [int(N_in/2**j) for j in range(3)]
            
        else:
            N_layers = [None for j in range(3)]

        self.layer1 = self._make_layer(block, layer_channels[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size, N_aps = N_layers[0])
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size, N_aps = N_layers[1])
        self.layer4 = self._make_layer(block, layer_channels[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size, N_aps = N_layers[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_channels[3] * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1, N_aps = None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = [ApsPool(filt_size=filter_size, stride=stride, channels=self.inplanes, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion, return_poly_indices = False, N = N_aps),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size, conv_pad_type = self.conv_pad_type, apspool_criterion = self.apspool_criterion, N = N_aps))    
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size, conv_pad_type = self.conv_pad_type, apspool_criterion = self.apspool_criterion, N = N_aps))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
            
def resnet18(pretrained=False, filter_size=4, pool_only=False, apspool_criterion = 'l2', N_in = None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet18-aps model')
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], filter_size=filter_size, pool_only=pool_only, apspool_criterion = apspool_criterion, N_in = N_in, **kwargs)
    
    if pretrained == True:
        raise Exception('Pre-trained model currently not available.')
    
    return model



