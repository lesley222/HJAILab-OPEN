""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

BIAS = False


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
        )
        # for m in self.single_conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.single_conv(x)

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvRelu, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # for m in self.single_conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.single_conv(x)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # for m in self.single_conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.single_conv(x)

class ConvInRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvInRelu, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # for m in self.single_conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.single_conv(x)

class ConvSpRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvSpRelu, self).__init__()
        self.single_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = ConvBnRelu(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UpConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
        super(UpConvBnRelu, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            # nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)

class UpConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
        super(UpConvRelu, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS)

    def forward(self, x):
        return self.up(x)



