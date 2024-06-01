""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

BIAS = False


class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, pool=False):
        super(OneConv, self).__init__()
        self.pool = pool
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=BIAS),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2),
        )
        if pool:
            self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.single_conv(x)
        if self.pool:
            x = self.max_pool(x)
        return x


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=BIAS),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=BIAS),
            # nn.BatchNorm2d(out_channels),
            # # nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2),
        )
        for m in self.single_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=BIAS),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=BIAS),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        # for m in self.double_conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, bias=BIAS)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class VaeDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_down=False):
        super().__init__()
        if conv_down:
            self.maxpool_conv = nn.Sequential(
                SingleConv(in_channels, out_channels, stride=2),
                # DoubleConv(out_channels, out_channels)
                SingleConv(out_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                # DoubleConv(out_channels, out_channels)
                SingleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SingleConv(in_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)

class VaeUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, ext_channel=2, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=BIAS)

        # self.conv = DoubleConv(in_channels, out_channels)
        self.conv = SingleConv(in_channels+ext_channel, out_channels)

    def forward(self, x1, x2=None):
        # print(x1.size())
        x1 = self.up(x1)
        if x2 is None:
            return self.conv(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=BIAS)
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.conv(x)
