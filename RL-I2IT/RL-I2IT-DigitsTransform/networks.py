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
    def __init__(self, in_channels, dim_z, conv_down=False):
        super().__init__()
        self.dim_z = dim_z
        self.conv_in = SingleConv(in_channels, 16, stride=1)
        self.down1 = VaeDown(16, 32, conv_down)  # 28 -> 14
        self.down2 = VaeDown(32, 32, conv_down)  # 14 -> 7

        self.fc = OutConv(32, 2)

    def dist_multivar_normal(self, mu, logvar):
        var = logvar.exp() # 协方差对角 var >=0
        cov_matrix = torch.diag_embed(var) # 生成正定协方差（对角）矩阵
        # print(mu.shape, var.shape)
        dist = torch.distributions.MultivariateNormal(mu, cov_matrix)
        return dist

    def forward(self, x):
        batch = x.size(0)
        enc = [x] # 2, 28
        raw = x.clone().detach()
        x = self.conv_in(x)
        enc.append(x) # 14
        x = self.down1(x)
        enc.append(x) # 7
        x = self.down2(x)

        x = self.fc(x)
        x = x.view(batch, -1)

        mu, logvar = x.chunk(2, dim=1)
        logvar = logvar.clamp(-20, 2)

        dist = self.dist_multivar_normal(mu, logvar)

        return dist, enc


class Decoder(nn.Module):
    def __init__(self, dim_z):
        super().__init__()
        self.dim_z = dim_z
        self.fc = SingleConv(1, 32)
        self.up1 = UpSampling(32, 32) # 7 -> 14
        self.up2 = UpSampling(32+32, 32) # 14 -> 28
        self.conv1 = SingleConv(32+16, 16)
        self.out = OutConv(16+2, 2)

    def forward(self, x, enc):
        batch = x.size(0)
        x = x.view(batch, 1, 7, 7)
        x = self.fc(x)

        x = self.up1(x) # 14
        x = torch.cat([x, enc[-1]], dim=1)
        x = self.up2(x) # 28
        x = torch.cat([x, enc[-2]], dim=1)
        x = self.conv1(x)
        x = torch.cat([x, enc[-3]], dim=1)
        x = self.out(x)

        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, in_channels, nf=32):
        super(Critic, self).__init__()
        self.in_channels = in_channels
        self.nf = nf

        self.conv1 = OneConv(in_channels, nf*1, stride=1) # 28
        self.conv2 = OneConv(nf*1, nf*2, pool=True) # 28 -> 14
        self.conv3 = OneConv(nf*2, nf*2, pool=True) # 14 -> 7
        self.conv6 = OneConv(nf*2+1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(nf*2+1, 1)

    def forward(self, x, latent):
        batch = x.size(0)
        latent = latent.view(batch, 1, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([x, latent], dim=1)
        x = self.conv6(x)
        x = self.pool(x)
        value = x.view(batch, 1)
        # value = self.fc(x)

        return value



class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()
        if isinstance(size, int):
            size = (size, size)
        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)





















