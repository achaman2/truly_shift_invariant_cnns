#Code originally from https://github.com/adobe/antialiased-cnns

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


from models.lpf_models.blurpool import BlurPool
from models.circular_pad_layer import circular_pad


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf2-6e2ee76f.pth',
    'resnet18_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf3-449351b9.pth',
    'resnet18_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf4-8c77af40.pth',
    'resnet18_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf5-c1eed0a1.pth',
    'resnet34_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf2-4707aed9.pth',
    'resnet34_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf3-16aa6c48.pth',
    'resnet34_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf4-55747267.pth',
    'resnet34_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf5-85283561.pth',
    'resnet50_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf2-f0f7589d.pth',
    'resnet50_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf3-a4e868d2.pth',
    'resnet50_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4-994b528f.pth',
    'resnet50_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf5-9953c9ad.pth',
    'resnet101_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf2-3d00941d.pth',
    'resnet101_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf3-928f1444.pth',
    'resnet101_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf4-f8a116ff.pth',
    'resnet101_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf5-1f3745af.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    
    return nn.Sequential(circular_pad([1, 1, 1, 1]),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=0, groups=groups, bias=False)
                            )

def conv7x7(in_planes, out_planes, stride=1, groups=1):
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if(stride==1):
            self.conv2 = conv3x3(planes,planes)
        else:
            self.conv2 = nn.Sequential(BlurPool(planes, filt_size=filter_size, stride=stride),
                conv3x3(planes, planes),)
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

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups=groups)  # stride moved
        self.bn2 = norm_layer(width)
        if(stride==1):
            self.conv3 = conv1x1(width, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(BlurPool(width, filt_size=filter_size, stride=stride),
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

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, filter_size=1, pool_only=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.base_width = width_per_group

        if(pool_only):
            self.conv1 = conv7x7(3, self.inplanes, stride=2)
        else:
            self.conv1 = conv7x7(3, self.inplanes, stride=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if(pool_only):
            self.maxpool = nn.Sequential(*[circular_pad([0, 1, 0 ,1]), nn.MaxPool2d(kernel_size=2, stride=1), 
                BlurPool(self.inplanes, filt_size=filter_size, stride=2,)])
        else:
            self.maxpool = nn.Sequential(*[BlurPool(self.inplanes, filt_size=filter_size, stride=2,),
                                           circular_pad([0, 1, 0 ,1]),
                nn.MaxPool2d(kernel_size=2, stride=1), 
                BlurPool(self.inplanes, filt_size=filter_size, stride=2,)])

        self.layer1 = self._make_layer(block, 64, layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [BlurPool(filt_size=filter_size, stride=stride, channels=self.inplanes),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, base_width=self.base_width, norm_layer=norm_layer, filter_size=filter_size))

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


def resnet18(pretrained=False, filter_size=4, pool_only=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    print('Loading Resnet18-lpf model')
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model currently available')    
    return model


def resnet34(pretrained=False, filter_size=4, pool_only=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model currently available')    
    return model


def resnet50(pretrained=False, filter_size=4, pool_only=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model available')    
    return model


def resnet101(pretrained=False, filter_size=4, pool_only=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model currently available')    

    return model


def resnet152(pretrained=False, filter_size=4, pool_only=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        filter_size (int): Antialiasing filter size
        pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model currently available')    
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, filter_size=4, pool_only=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model currently available')    
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, filter_size=4, pool_only=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, filter_size=filter_size, pool_only=pool_only, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model currently available')    
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model