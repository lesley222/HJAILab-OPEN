import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from convGRU import ConvGRU
from layers import *

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
epsilon = 1e-6


#  Initialize Policy weights
# def conv_weights_init_(m):
#     if isinstance(m, nn.Conv2d):
#         torch.nn.init.kaiming_normal_(m.weight)

def linear_weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, state_input_dim, action_input_dim, hidden_dim=16):
        super(QNetwork, self).__init__()

        # state extract                                                     [n, 3, 256, 256]
        self.conv1_s = ConvRelu(state_input_dim, hidden_dim, 4, 2, 1)  # [n, hd, 128, 128]
        self.conv2_s = ConvInRelu(hidden_dim, hidden_dim, 4, 2, 1)  # [n, hd, 64, 64]
        self.conv3_s = ConvInRelu(hidden_dim, hidden_dim * 2, 4, 2, 1)  # [n, hd*2, 32, 32]
        self.conv4_s = ConvInRelu(hidden_dim * 2, hidden_dim * 2, 4, 2, 1)  # [n, hd*2, 16, 16]
        self.conv5_s = ConvInRelu(hidden_dim * 2, hidden_dim * 4, 4, 2, 1)  # [n, hd*4, 8, 8]
        self.conv6_s = ConvInRelu(hidden_dim * 4, hidden_dim * 4, 4, 2, 1)  # [n, hd*4, 4, 4]
        self.conv7_s = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, (4, 4), (1, 1), bias=False)  # [n, hd*8, 1, 1]

        # action extract                                                      [n, 64, 64, 64]
        self.conv1_a = ConvRelu(action_input_dim, hidden_dim, 4, 2, 1)  # [n, hd, 32, 32]
        self.conv2_a = ConvInRelu(hidden_dim, hidden_dim, 4, 2, 1)  # [n, hd, 16, 16]
        self.conv3_a = ConvInRelu(hidden_dim, hidden_dim * 2, 4, 2, 1)  # [n, hd*2, 8, 8]
        self.conv4_a = ConvInRelu(hidden_dim * 2, hidden_dim * 4, 4, 2, 1)  # [n, hd*4, 4, 4]
        self.conv5_a = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, (4, 4), (1, 1), bias=False)  # [n, hd*8, 1, 1]

        # Q1 architecture
        self.linear1 = nn.Linear(hidden_dim * 8 + hidden_dim * 8, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(hidden_dim * 8 + hidden_dim * 8, hidden_dim * 4)
        self.linear5 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        # self.apply(conv_weights_init_)
        self.apply(linear_weights_init_)

    def forward(self, state, action):
        num_batch = state.size()[0]
        s = self.conv1_s(state)
        s = self.conv2_s(s)
        s = self.conv3_s(s)
        s = self.conv4_s(s)
        s = self.conv5_s(s)
        s = self.conv6_s(s)
        s = self.conv7_s(s)
        s_ = s.view(num_batch, -1)
        a = self.conv1_a(action)
        a = self.conv2_a(a)
        a = self.conv3_a(a)
        a = self.conv4_a(a)
        a = self.conv5_a(a)
        a_ = a.view(num_batch, -1)

        xu = torch.cat([s_, a_], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, action_dim, hidden_dim=16):
        super(GaussianPolicy, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.conv1 = ConvLayer(num_inputs, hidden_dim, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(hidden_dim, affine=True)

        self.conv2 = ConvLayer(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(hidden_dim * 2, affine=True)

        self.conv3 = ConvLayer(hidden_dim * 2, action_dim, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(action_dim, affine=True)

        self.gru_step = ConvGRU(action_dim, action_dim, 3, 1)

        self.mean_conv = ConvLayer(action_dim, action_dim, kernel_size=3, stride=1)
        self.inm_e = nn.InstanceNorm2d(action_dim, affine=True)

        self.log_std_conv = ConvLayer(action_dim, action_dim, kernel_size=3, stride=1)
        self.inl_e = nn.InstanceNorm2d(action_dim, affine=True)

        self.step_hidden_state = None
        # self.apply(conv_weights_init_)

    def forward(self, state):
        enc = [state]
        x = self.relu(self.in1_e(self.conv1(state)))
        enc.append(x)
        x = self.relu(self.in2_e(self.conv2(x)))
        enc.append(x)
        x = self.relu(self.in3_e(self.conv3(x)))
        enc.append(x)  # [n, frames, h, w]
        x = x.unsqueeze(0)
        hidden_state = self.gru_step(x, self.step_hidden_state)[0]  # [seq_len, batch, channels, height, width]
        x = hidden_state[0]
        self.step_hidden_state = hidden_state.detach()
        mean = self.relu(self.inm_e(self.mean_conv(x)))
        log_std = self.tanh(self.inl_e(self.mean_conv(x)))

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std, enc

    def sample(self, state):
        mean, log_std, enc = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean, enc

    def init_hidden_state(self):
        self.step_hidden_state = None


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, action_dim, hidden_dim=32):
        super(DeterministicPolicy, self).__init__()
        # nonlineraity
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(num_inputs, hidden_dim, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(hidden_dim, affine=True)

        self.conv2 = ConvLayer(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(hidden_dim * 2, affine=True)

        self.conv3 = ConvLayer(hidden_dim * 2, action_dim, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(action_dim, affine=True)

        # residual layers
        self.res1 = ResidualBlock(action_dim)
        self.res2 = ResidualBlock(action_dim)
        self.res3 = ResidualBlock(action_dim)
        self.res4 = ResidualBlock(action_dim)
        self.res5 = ResidualBlock(action_dim)

        # self.apply(conv_weights_init_)

    def forward(self, state):
        # encode
        y = self.relu(self.in1_e(self.conv1(state)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        return y

    def sample(self, state):
        mean = self.forward(state)
        # self.noise = torch.normal(mean=0., std=0.1, size=mean.size())
        # self.noise = self.noise.clamp(-0.25, 0.25)
        # action = mean + self.noise
        action = mean
        return action, torch.zeros_like(action), mean


class Executor(nn.Module):
    def __init__(self, action_dim, hidden_dim, output_dim=3):
        super(Executor, self).__init__()

        # nonlinear
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # decoding layers
        self.deconv3 = UpsampleConvLayer(action_dim + 64, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64 + 32, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32 + 16, output_dim, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(output_dim, affine=True)

        # self.apply(conv_weights_init_)

    def forward(self, y, enc):
        y = torch.cat([y, enc[-1]], dim=1)
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = torch.cat([y, enc[-2]], dim=1)
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = torch.cat([y, enc[-3]], dim=1)
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y


class Echo(nn.Module):
    def __init__(self, action_dim=64, max_episode_step=10):
        super(Echo, self).__init__()
        self.memory_size = max_episode_step
        self.gru = ConvGRU(action_dim, action_dim, 3, 1)
        self.frame_hidden_state = [None] * max_episode_step

    def forward(self, x, step):
        x = x.unsqueeze(0)
        out = self.gru(x, self.frame_hidden_state[step])[0]
        self.frame_hidden_state[step] = out.detach()
        x = out[-1]
        return x

    def init_hidden_state(self):
        self.frame_hidden_state = [None]*self.memory_size


if __name__ == "__main__":
    net = Echo()  # 221376
    pnet = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pnet)
