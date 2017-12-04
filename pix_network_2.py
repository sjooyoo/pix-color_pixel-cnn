import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


#input = 224 x 224 x 3

class UNetConvBlock1(nn.Module):
    # feature map 64
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock1, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=2, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock2(nn.Module):
    # feature map 128
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock2, self).__init__()
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, 3, stride=2, groups=out_size, bias=False)
        #self.conv3.weight.data.fill_(1)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        out = self.conv3(out)
        return out

class UNetConvBlock3(nn.Module):
    # feature map 256
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock3, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size, stride=2, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock4(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock4, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        #self.conv3.weight.data.fill_(1)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock5(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock5, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock6(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        # feature maps 512
        super(UNetConvBlock6, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, kernel_size, groups=out_size, bias=False)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        #self.conv4.weight.data.fill_(1)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock7(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=5, activation=F.relu):
        #feature maps 1024
        super(UNetConvBlock7, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock8(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        # feature map 512
        super(UNetConvBlock8, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock9(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        # feature map 128
        super(UNetConvBlock9, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


# bilinear upsampling
class UNetConvBlock10(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        # feature map 64
        super(UNetConvBlock10, self).__init__()
        self.up = nn.UpsamplingBilinear2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.up(x)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)

        return out


# bilinear upsampling
class UNetConvBlock11(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        # feature map 32
        super(UNetConvBlock11, self).__init__()
        self.up = nn.UpsamplingBilinear2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.up(x)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out

class UNetConvBlock12(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock12, self).__init__()
        self.conv = nn.Conv2d(out_size, out_size, kernel_size=1, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
    def forward(self, x, bridge):
        out = self.activation(x)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)
        return out


class refinement_network(nn.Module):
    def __init__(self, imsize):
        super(refinement_network, self).__init__()
        self.imsize = imsize

        self.convlayer1 = UNetConvBlock1(1, 64)
        self.convlayer2 = UNetConvBlock2(64, 128)
        self.convlayer3 = UNetConvBlock3(128, 256)
        self.convlayer4 = UNetConvBlock4(256, 512)
        self.convlayer5 = UNetConvBlock5(512, 256)
        self.convlayer6 = UNetConvBlock6(256, 512)
        self.convlayer7 = UNetConvBlock7(512, 1024)
        self.convlayer8 = UNetConvBlock8(1024, 512)
        self.convlayer9 = UNetConvBlock9(512, 128)
        self.convlayer10 = UNetConvBlock10(128, 64)
        self.convlayer11 = UNetConvBlock11(64, 32)
        self.convlayer12 = UNetConvBlock12(32, 2)
        self.resize = transforms.Scale(imsize, interpolation=2)



    def forward(self, x1, x2):
        x2 = self.resize(x2)
        out = torch.cat(x1, x2)
        layer1 = self.convlayer1(out)
        layer2 = self.convlayer2(layer1)
        layer3 = self.convlayer3(layer2)
        layer4 = self.convlayer4(layer3)
        layer5 = self.convlayer5(layer4)
        layer6 = self.convlayer6(layer5)
        layer7 = self.convlayer7(layer6)
        layer8 = self.convlayer8(layer7)
        layer9 = self.convlayer9(layer8)
        layer10 = self.convlayer10(layer9)
        layer11 = self.convlayer11(layer10)
        layer12 = self.convlayer12(layer11)
        out = layer12

        return out
















