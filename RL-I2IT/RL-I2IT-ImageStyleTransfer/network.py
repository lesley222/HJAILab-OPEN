import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


logvar_max=2
logvar_min=-10
def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c, hight, width = features.size()
    mu_block, logvar_block = features.chunk(2, dim=1)

    logvar_block = torch.tanh(logvar_block)  # 变换到(-1,1)区间
    logvar_block = logvar_min + 0.5 * (
            logvar_max - logvar_min
    ) * (logvar_block + 1)

    var_block = logvar_block.exp()
    features_mean = mu_block.mean(dim=1).reshape(batch_size, -1, hight, width)
    features_std = var_block.std(dim=1).reshape(batch_size, -1, hight, width) + 1e-6

    return features_mean, features_std


# import distribution
def dist_multivar_normal(mu, var):   # mu.size() [1, 1, 64, 64]  var.size() [1, 1, 64, 64]
    dist = torch.distributions.normal.Normal(mu, var)
    return dist


def sample(features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """

    mean, std = calc_mean_std(features)
    dist = dist_multivar_normal(mean, std)
    return dist


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(16, affine=True)

        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(32, affine=True)

        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(64, affine=True)

        # residual layers
        self.res1 = ResidualBlock(64)
        # self.res2 = ResidualBlock(128)
        # self.res3 = ResidualBlock(128)
        # self.res4 = ResidualBlock(128)
        # self.res5 = ResidualBlock(128)

    def forward(self, x):
        # encode
        enc = [x]
        y = self.relu(self.in1_e(self.conv1(x)))  # [n, 32, 256, 256]
        enc.append(y)
        y = self.relu(self.in2_e(self.conv2(y)))  # [n, 64, 128, 128]
        enc.append(y)
        y = self.relu(self.in3_e(self.conv3(y)))  # [n, 128, 64, 64]
        enc.append(y)

        # residual layers
        y = self.res1(y)

        dist = sample(y)

        return dist, enc


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # nonlinear
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # decoding layers
        self.deconv3 = UpsampleConvLayer(1+64, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64+32, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32+16, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, y, enc):
        # decode
        y = torch.cat([y, enc[-1]], dim=1)
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = torch.cat([y, enc[-2]], dim=1)
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = torch.cat([y, enc[-3]], dim=1)
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y


class Critic(nn.Module):
    def __init__(self, n_channel=3, nf=64, bottle=64*64):
        super(Critic, self).__init__()
        self.conv1 = ConvRelu(n_channel, nf * 1, 4, 2, 1)  # [n, 3, 256, 256] -> [n, 64, 128, 128]
        self.conv2 = ConvInRelu(nf * 1, nf * 1, 4, 2, 1)  # [n, 64, 128, 128] -> [n, 64, 64, 64]
        self.conv3 = ConvInRelu(nf * 1 + 1, nf * 2, 4, 2, 1)  # [n, 64+1, 64, 64] -> [n, 128, 32, 32]
        self.conv4 = ConvInRelu(nf * 2, nf * 4, 4, 2, 1)  # [n, 128, 32, 32] -> [n, 256, 16, 16]
        self.conv5 = ConvInRelu(nf * 4, nf * 8, 4, 2, 1)  # [n, 256, 16, 16] -> [n, 512, 8, 8]
        self.conv6 = ConvInRelu(nf * 8, nf * 16, 4, 2, 1)  # [n, 512, 8, 8] -> [n, 1024, 4, 4]
        self.conv7 = Conv(nf * 16, bottle, 4, 1, 0)  # [n, 1024, 4, 4] -> [n, 512, 1, 1]
        self.norm = nn.BatchNorm2d(64+1)
        self.fc = nn.Linear(bottle, 64*64)

    def forward(self, x, latent):  # [n, 3, 256, 256]
        batch = x.size(0)
        x = self.conv1(x)  # [n, 64, 128, 128]
        x = self.conv2(x)  # [n, 64, 64, 64]
        x = torch.cat([x, latent], dim=1)  # [n, 65, 64, 64]
        x = F.leaky_relu(self.norm(x), 0.2)  # [n, 65, 64, 64]
        x = self.conv3(x)  # [n, 128, 32, 32]
        x = self.conv4(x)  # [n, 256, 16, 16]
        x = self.conv5(x)  # [n, 512, 8, 8]
        x = self.conv6(x)  # [n, 1024, 4, 4]
        x = self.conv7(x).view(batch, -1)  # [n, 512]
        value = self.fc(x).view(batch, -1, 64, 64)

        return value


class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(16, affine=True)

        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(32, affine=True)

        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(64, affine=True)

        # residual layers
        self.res1 = ResidualBlock(64)

        # nonlinear
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # decoding layers
        self.deconv3 = UpsampleConvLayer(1 + 64, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64 + 32, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32 + 16, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        enc = [x]
        y = self.relu(self.in1_e(self.conv1(x)))  # [n, 32, 256, 256]
        enc.append(y)
        y = self.relu(self.in2_e(self.conv2(y)))  # [n, 64, 128, 128]
        enc.append(y)
        y = self.relu(self.in3_e(self.conv3(y)))  # [n, 128, 64, 64]
        enc.append(y)

        # residual layers
        y = self.res1(y)

        dist = sample(y)
        y = dist.sample()

        y = torch.cat([y, enc[-1]], dim=1)
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = torch.cat([y, enc[-2]], dim=1)
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = torch.cat([y, enc[-3]], dim=1)
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y


if __name__ == "__main__":
    net = Critic()
    pnet = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pnet)
