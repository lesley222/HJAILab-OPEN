import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import cv2

from config import Config as cfg
from layers import *
import utils


class Encoder(nn.Module):
    def __init__(self, nc=3, nf=32, bottle=512, logvar_max=2, logvar_min=-10):
        super().__init__()
        self.bottle = bottle
        self.logvar_max = logvar_max
        self.logvar_min = logvar_min

        self.down1 = ConvRelu(nc, nf, 4, 2, 1) # 128 -> 64
        self.down2 = ConvBnRelu(nf, nf, 4, 2, 1)  # 64 -> 32
        self.down3 = ConvBnRelu(nf, nf*2, 4, 2, 1)  # 32 -> 16
        self.down4 = ConvBnRelu(nf*2, nf*4, 4, 2, 1) # 16  -> 8
        self.down5 = ConvBnRelu(nf*4, nf*8, 4, 2, 1) # 8  -> 4
        # self.down6 = ConvBnRelu(nf*8, nf*8, 4, 2, 1) # 8  -> 4
        self.out = Conv(nf*8, bottle*2, 4, 1, 0) # 1

    def dist_multivar_normal(self, mu, logvar):
        var = logvar.exp() # 协方差对角 var >=0
        cov_matrix = torch.diag_embed(var) # 生成正定协方差（对角）矩阵
        dist = torch.distributions.MultivariateNormal(mu, cov_matrix)
        return dist

    def forward(self, x):
        batch = x.size(0)
        enc = [x] # 2, 128
        x = self.down1(x)
        enc.append(x) # 16, 64
        x = self.down2(x)
        enc.append(x) # 64, 32
        x = self.down3(x)
        enc.append(x) # 32, 16
        x = self.down4(x)
        enc.append(x) # 16, 8
        x = self.down5(x)
        enc.append(x) # 8, 4

        x = self.out(x) # 1x1xbottle*2
        x = x.view(batch, -1)
        mu, logvar = x.chunk(2, dim=1)

        logvar = torch.tanh(logvar)
        logvar = self.logvar_min + 0.5 * (
            self.logvar_max - self.logvar_min
        ) * (logvar + 1)

        # logvar = logvar.clamp(self.logvar_min, self.logvar_max)

        dist = self.dist_multivar_normal(mu, logvar)

        return dist, enc


class Decoder(nn.Module):
    def __init__(self, nf, bottle):
        super().__init__()
        self.bottle = bottle
        self.up1 = UpSampling(bottle, nf*8, 4)
        self.up2 = UpSampling(nf*8 + nf*8, nf*8)
        self.up3 = UpSampling(nf*8 + nf*4, nf*4)
        self.up4 = UpSampling(nf*4 + nf*2, nf*2)
        self.up5 = UpSampling(nf*2 + nf*1, nf*1)
        self.up6 = UpSampling(nf*1 + nf*1, nf*1)
        # self.up7 = UpSampling(nf*1 + nf*1, nf)

        self.out = Conv(nf, 3, 3, 1, 1)

    def forward(self, x, enc):
        batch = x.size(0)
        x = x.view(batch, self.bottle, 1, 1)
        x = self.up1(x) # 4
        x = torch.cat([x, enc[-1]], dim=1)
        x = self.up2(x) # 8
        x = torch.cat([x, enc[-2]], dim=1)
        x = self.up3(x) # 16
        x = torch.cat([x, enc[-3]], dim=1)
        x = self.up4(x) # 32
        x = torch.cat([x, enc[-4]], dim=1)
        x = self.up5(x) # 64
        x = torch.cat([x, enc[-5]], dim=1)
        x = self.up6(x) # 128
        # x = torch.cat([x, enc[-6]], dim=1)
        # x = self.up7(x) # 256

        # x = self.out(x)
        x = torch.tanh(self.out(x))

        # return torch.clamp(x, -1, 1)
        return x


class Critic(nn.Module):
    def __init__(self, nc, nf=32, bottle=512):
        super(Critic, self).__init__()
        self.conv1 = ConvRelu(nc, nf*1, 4, 2, 1) # 128 -> 64
        self.conv2 = ConvBnRelu(nf*1, nf*1, 4, 2, 1) # 64 -> 32
        self.conv3 = ConvBnRelu(nf*1, nf*2, 4, 2, 1) # 32 -> 16
        self.conv4 = ConvBnRelu(nf*2, nf*4, 4, 2, 1) # 16 > 8
        self.conv5 = ConvBnRelu(nf*4, nf*8, 4, 2, 1) # 8 -> 4
        # self.conv6 = ConvBnRelu(nf*8, nf*8, 4, 2, 1) # 8 -> 4
        self.conv7 = Conv(nf*8, bottle, 4, 1, 0) # 1
        self.norm = nn.BatchNorm1d(bottle*2)
        self.fc = nn.Linear(bottle*2, 1)

    def forward(self, x, latent):
        batch = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.conv7(x).view(batch, -1)
        x = torch.cat([x, latent], dim=1)
        x = F.leaky_relu(self.norm(x), 0.2)
        value = self.fc(x)

        return value


class NetD(nn.Module):
    def __init__(self, nc, nf=32):
        super(NetD, self).__init__()
        self.main = nn.Sequential(
            ConvRelu(nc, nf*1, 4, 2, 1), # 128 -> 64
            ConvSpRelu(nf*1, nf*1, 4, 2, 1), # 32 -> 32
            ConvSpRelu(nf*1, nf*2, 4, 2, 1), # 16 -> 16
            ConvSpRelu(nf*2, nf*4, 4, 2, 1), # 8 -> 8
            ConvSpRelu(nf*4, nf*8, 4, 2, 1), # 8 -> 4
            # ConvSpRelu(nf*8, nf*8, 4, 2, 1), # 8 -> 4
        )
        self.out = nn.Sequential(
            Conv(nf*8, 1, 4, 1, 0), # 4 -> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.main(x)
        out = self.out(features).view(-1, 1)
        return out









