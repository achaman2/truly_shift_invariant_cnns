
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_classes.models_for_cifar10.aps_models.apspool import ApsPool, polyphase_comp_select

from model_classes.models_for_cifar10.circular_pad_layer import circular_pad


def conv3x3(in_planes, out_planes, stride=1, groups=1, conv_pad_type = 'zeros'):
    """3x3 convolution with padding"""
    
    if conv_pad_type == 'zeros':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)
    
    elif conv_pad_type == 'reflect':
        return nn.Sequential(nn.ReflectionPad2d([1, 1, 1, 1]),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=0, groups=groups, bias=False)
                            )
    
    elif conv_pad_type == 'replicate_pad':
        return nn.Sequential(nn.ReplicationPad2d([3, 3, 3, 3]),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=0, groups=groups, bias=False)
                            )
    
    elif conv_pad_type == 'circular':
#    
       return nn.Sequential(circular_pad([1, 1, 1, 1]),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=0, groups=groups, bias=False)
                            )
        

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, filter_size=1, conv_pad_type = 'zeros', apspool_criterion = 'l2'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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
        
        
        if(stride==1):
            self.conv2 = conv3x3(planes,planes, conv_pad_type = conv_pad_type)
        else:
            self.conv2 = nn.Sequential(ApsPool(planes, filt_size=filter_size, stride=stride, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion),
                conv3x3(planes, planes, conv_pad_type = conv_pad_type),)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.stride >1:
            indices = polyphase_comp_select(out, apspool_criterion = self.apspool_criterion)
            out = self.conv2({'inp': out, 'polyphase_indices': indices})
        
        else:
            out = self.conv2(out)
        
        out = self.bn2(out)

        if self.downsample is not None:
            if self.stride >1:
                identity = self.downsample({'inp': x, 'polyphase_indices': indices})
            else:
                identity = self.downsample(x)
                
        N_out = out.shape[2]
        N_id = identity.shape[2]
        
        if N_id!=N_out:
#             this would only be needed if replicate mode with large padding is used in conv layers
            pad_extra = N_out - N_id
            identity = F.pad(identity, (0, pad_extra, 0, pad_extra) , mode = 'replicate')
            
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, filter_size=1, conv_pad_type = 'zeros', apspool_criterion = 'l2'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups
        self.circular_flag = False
        if conv_pad_type == 'circular':
            self.circular_flag = True
            
        self.apspool_criterion = apspool_criterion
        
#         if apspool_criterion !='l2':
            
            
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups=groups, conv_pad_type = conv_pad_type) # stride moved
        self.bn2 = norm_layer(width)
        if(stride==1):
            self.conv3 = conv1x1(width, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(ApsPool(width, filt_size=filter_size, stride=stride, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion),
                conv1x1(width, planes * self.expansion))
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
            indices = polyphase_comp_select(out, apspool_criterion = self.apspool_criterion)
            out = self.conv3({'inp': out, 'polyphase_indices': indices})
        else:
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


class Small_ResNet(nn.Module):
# small resnet one less (res layer) than the regular resnet with channels of the order [16, 32, 64]
# this is based on the implementation used by original resnet paper for cifar10

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, layer_channels = [16, 32, 64], width_per_group=64, norm_layer=None, filter_size=1, 
                 pool_only=True, conv_pad_type = 'zeros', apspool_criterion = 'l2'):
        
        super(Small_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = layer_channels[0]
        self.base_width = width_per_group
        self.conv_pad_type = conv_pad_type
        self.circular_flag = False
        if conv_pad_type == 'circular':
            self.circular_flag = True
            
        self.apspool_criterion = apspool_criterion 
            

        self.conv1 = conv3x3(3, self.inplanes, stride = 1, conv_pad_type = conv_pad_type)
                        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, layer_channels[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_channels[2] * block.expansion, num_classes)
        
        
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

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = [ApsPool(filt_size=filter_size, stride=stride, channels=self.inplanes, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size, conv_pad_type = self.conv_pad_type, apspool_criterion = self.apspool_criterion))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size, conv_pad_type = self.conv_pad_type, apspool_criterion = self.apspool_criterion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, layer_channels = [64, 128, 256, 512], width_per_group=64, norm_layer=None, filter_size=1, 
                 pool_only=True, conv_pad_type = 'zeros', apspool_criterion = 'l2'):
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

        
        self.conv1 = conv3x3(3, self.inplanes, stride = 1, conv_pad_type = conv_pad_type)
                        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, layer_channels[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer4 = self._make_layer(block, layer_channels[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        
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

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = [ApsPool(filt_size=filter_size, stride=stride, channels=self.inplanes, circular_flag = self.circular_flag, apspool_criterion = self.apspool_criterion),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size, conv_pad_type = self.conv_pad_type, apspool_criterion = self.apspool_criterion))    
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size, conv_pad_type = self.conv_pad_type, apspool_criterion = self.apspool_criterion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_aps(pretrained=False, filter_size=4, layer_channels = [16, 32, 64], pool_only=True, conv_pad_type = 'zeros', dataset_to_train = 'cifar10', apspool_criterion = 'l2', **kwargs):
    """Constructs a ResNet-20 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet20_aps\n')
    model = Small_ResNet(BasicBlock, [3, 3, 3], filter_size=filter_size, layer_channels = layer_channels, 
                   pool_only=pool_only, conv_pad_type = conv_pad_type, apspool_criterion = apspool_criterion, **kwargs)
    
    if pretrained:
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train == 'imagenet':
        raise ValueError('Only resnets designed for cifar10 (small kernel size in first layer) can be loaded here')

    return model


def resnet56_aps(pretrained=False, filter_size=4, layer_channels = [16, 32, 64], pool_only=True, conv_pad_type = 'zeros', dataset_to_train = 'cifar10', apspool_criterion = 'l2', **kwargs):
    """Constructs a ResNet-56 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet56_aps\n')
    model = Small_ResNet(BasicBlock, [9, 9, 9], filter_size=filter_size, layer_channels = layer_channels, 
                   pool_only=pool_only, conv_pad_type = conv_pad_type, apspool_criterion = apspool_criterion, **kwargs)
    
    if pretrained:
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train == 'imagenet':
        raise ValueError('Only resnets designed for cifar10 (small kernel size in first layer) can be loaded here')

    return model


def resnet18_aps(pretrained=False, filter_size=4, layer_channels = [64, 128, 256, 512], pool_only=True, conv_pad_type = 'zeros', dataset_to_train = 'cifar10', apspool_criterion = 'l2', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet18_aps\n')
    model = ResNet(BasicBlock, [2, 2, 2, 2], filter_size=filter_size, layer_channels = layer_channels, 
                   pool_only=pool_only, conv_pad_type = conv_pad_type, apspool_criterion = apspool_criterion,**kwargs)
    
    if pretrained:
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train == 'imagenet':
        raise ValueError('Only resnets designed for cifar10 (small kernel size in first layer) can be loaded here')

    return model


def resnet34_aps(pretrained=False, filter_size=4, layer_channels = [64, 128, 256, 512], pool_only=True, conv_pad_type = 'zeros', dataset_to_train = 'cifar10', apspool_criterion = 'l2', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet34_aps\n')
    model = ResNet(BasicBlock, [3, 4, 6, 3], filter_size=filter_size, layer_channels = layer_channels, 
                   pool_only=pool_only, conv_pad_type = conv_pad_type, apspool_criterion = apspool_criterion,**kwargs)
    
    if pretrained:
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train == 'imagenet':
        raise ValueError('Only resnets designed for cifar10 (small kernel size in first layer) can be loaded here')

    return model



def resnet50_aps(pretrained=False, filter_size=4, layer_channels = [64, 128, 256, 512], pool_only=True, conv_pad_type = 'zeros', dataset_to_train = 'cifar10', apspool_criterion = 'l2', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet50_aps\n')
    model = ResNet(Bottleneck, [3, 4, 6, 3], filter_size=filter_size, layer_channels = layer_channels, 
                   pool_only=pool_only, conv_pad_type = conv_pad_type, apspool_criterion = apspool_criterion,**kwargs)
    
    if pretrained:
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train == 'imagenet':
        raise ValueError('Only resnets designed for cifar10 (small kernel size in first layer) can be loaded here')

    return model



