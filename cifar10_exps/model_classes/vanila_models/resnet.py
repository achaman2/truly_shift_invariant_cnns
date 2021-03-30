# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, conv_pad_type = 'zeros'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        
        self.conv1 = conv3x3(inplanes, planes, stride, conv_pad_type = conv_pad_type)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_pad_type = conv_pad_type)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        N_out = out.shape[2]
        N_id = identity.shape[2]
        
        if N_id!=N_out:
#             this would only be needed when conv3x3 has replicate mode on with large padding
            pad_extra = N_out - N_id
            identity = F.pad(identity, (0, pad_extra, 0, pad_extra) , mode = 'replicate')
            

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation = 1, norm_layer=None, conv_pad_type = 'zeros'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride = stride, groups=groups, conv_pad_type = conv_pad_type) # stride moved
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        
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

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class Small_ResNet(nn.Module):
# small resnet -> one less (res layer) than the regular resnet with channels of the order [16, 32, 64]
# this is based on the implementation used by original resnet paper for cifar10

    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, layer_channels = [16, 32, 64], width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, conv_pad_type = 'zeros', ):
        
        super(Small_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        
        self.conv_pad_type = conv_pad_type
        self.num_classes = num_classes
        self.layer_channels = layer_channels
        
        self.inplanes = layer_channels[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = conv3x3(3, self.inplanes, stride=1, conv_pad_type = self.conv_pad_type)                       
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
       
        self.layer1 = self._make_layer(block, layer_channels[0], layers[0])
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_channels[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, conv_pad_type = self.conv_pad_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, conv_pad_type = self.conv_pad_type))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)
    

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, layer_channels = [64, 128, 256, 512], width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, conv_pad_type = 'zeros', ):
        
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        
        self.conv_pad_type = conv_pad_type
        self.num_classes = num_classes
        self.layer_channels = layer_channels
        
        self.inplanes = layer_channels[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = conv3x3(3, self.inplanes, stride=1, conv_pad_type = self.conv_pad_type)                       
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
       
        self.layer1 = self._make_layer(block, layer_channels[0], layers[0])
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.layer4 = self._make_layer(block, layer_channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_channels[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, conv_pad_type = self.conv_pad_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, conv_pad_type = self.conv_pad_type))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)
    
    
    
def _small_resnet(arch, block, layers, dataset_to_train, conv_pad_type, pretrained, progress, filter_size = 1, 
            layer_channels = [16, 32, 64], **kwargs):

#     note filter size 1 is not needed in this code
        
    model = Small_ResNet(block, layers,  num_classes = 10, layer_channels = layer_channels, conv_pad_type = conv_pad_type,
                 **kwargs)

    if pretrained == True:
        
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train != 'cifar10':
        
        raise ValueError('Only resnets designed for cifar10 (with small kernel size in first conv) can be loaded here')
        
        
    return model



def resnet20(dataset_to_train = 'cifar10', conv_pad_type = 'zeros', layer_channels = [16, 32, 64], pretrained=False, progress=True, **kwargs):
    r"""ResNet-20 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
    """
    print('\nLoading Resnet-20' )
    
    
    return _small_resnet('resnet20', BasicBlock, [3, 3, 3], dataset_to_train, conv_pad_type, pretrained, progress,
                   layer_channels = layer_channels, **kwargs)



def resnet56(dataset_to_train = 'cifar10', conv_pad_type = 'zeros', pretrained=False, progress=True, layer_channels = [16, 32, 64], **kwargs):
    r"""ResNet-56 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
    """
    print('\nLoading Resnet-56' )
    
    return _small_resnet('resnet56', BasicBlock, [9, 9, 9], dataset_to_train, conv_pad_type, pretrained, progress,
                   layer_channels = layer_channels, **kwargs)



def _resnet(arch, block, layers, dataset_to_train, conv_pad_type, pretrained, progress, filter_size = 1, 
            layer_channels = [64, 128, 256, 512], **kwargs):

#     note filter size 1 is not needed in this code
        
    model = ResNet(block, layers,  num_classes = 10, layer_channels = layer_channels, conv_pad_type = conv_pad_type,
                 **kwargs)

    if pretrained == True:
        
        raise ValueError('Pre-trained option not currently supported')
        
    if dataset_to_train != 'cifar10':
        
        raise ValueError('Only resnets designed for cifar10 can be loaded here')
        
        
    return model    
    
    
# ResNet(BasicBlock, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
def resnet18(dataset_to_train = 'cifar10', conv_pad_type = 'zeros', pretrained=False, progress=True, layer_channels = [64, 128, 256, 512], **kwargs):
    r"""ResNet-56 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
    """
    print('\nLoading Resnet-18' )

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], dataset_to_train, conv_pad_type, pretrained, progress,
                   layer_channels = layer_channels, **kwargs)


def resnet34(dataset_to_train = 'cifar10', conv_pad_type = 'zeros', pretrained=False, progress=True, layer_channels = [64, 128, 256, 512], **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
    """
    print('\nLoading Resnet-34' )

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], dataset_to_train, conv_pad_type, pretrained, progress,
                   layer_channels = layer_channels, **kwargs)

    
def resnet50(dataset_to_train = 'cifar10', conv_pad_type = 'zeros', pretrained=False, progress=True, layer_channels = [64, 128, 256, 512], **kwargs):
    r"""ResNet-56 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
    """
    print('\nLoading Resnet-50' )

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], dataset_to_train, conv_pad_type, pretrained, progress,
                   layer_channels = layer_channels, **kwargs)

    


    

# %%


# %%


# %%
