import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import shutil
import copy
import torchvision.utils as vutils
import scipy.misc
from skimage.metrics import structural_similarity as ssim_

from config import Config as cfg
import os
import cv2
import random
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # cpu\gpu 结果一致

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    return ssim_(img1, img2, multichannel=True)

def calculate_discount(r, bootstrap):
    size = len(r)
    R_batch = np.zeros(size, np.float32)
    R = bootstrap
    for i in reversed(range(0, size)):
        R = r[i] + cfg.GAMMA * R
        R_batch[i] = R

    return R_batch


def calculate_advantage(r, v, bootstrap):
    """
    Calulates the Advantage Funtion for each timestep t in the batch, where:
        - At = \sum_{i=0}^{k-1} gamma^{i} * r_{t+i} + gamma^{k} * V(s_{t+k}) - V(s_{t})
    where V(s_{t+k}) is the bootstraped value (if the episode hasn't finished).
    This results in a 1-step update for timestep B, 2-step update for timestep
    B-1,...., B-step discounted reward for timestep 1, where:
        - B = Batch Size
    Example: consider B = 3. Therefore, it results in the following advantage vector:
        - A[0] = r_1 + gamma * r_2 + gamma^2 * r_3 + gamma^3 * bootstrap - V(s_1)
        - A[1] = r_2 + gamma * r_3 + gamma^2 * bootstrap - V(s_2)
        - A[2] = r_3 + gamma * bootstrap - V(s_3)
    """
    size = len(r)
    A_batch = np.zeros([size], np.float32)
    aux = bootstrap
    for i in reversed(range(0, size)):
        aux = r[i] + cfg.GAMMA * aux
        A = aux - v[i]
        A_batch[i] = A

    return A_batch


def mkdir(path):
    # give a path, create the folder
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def deldir(path):
    folder = os.path.exists(path)
    if folder:
        shutil.rmtree(path)

def remkdir(path):
    deldir(path)
    mkdir(path)

def load_image(filename):
    im = Image.open(filename)
    return im.convert('RGB')

def resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def tensor_im(data):
    im = Image.fromarray(data).convert('L')
    return TF.to_tensor(im)

def tensor(data, dtype=torch.float32, device=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def numpy_im(data, device=None):
    im = (numpy(data, device).squeeze().transpose(1,2,0)+1) * 127.5
    return im

def numpy(data, device=None):
    if device is not None:
        return data.detach().cpu().numpy()
    else:
        return data.detach().numpy()

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def un_norm(tensors, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensors_ = tensors.clone()
    # for tensor in tensors_:
    #     for t, m, s in zip(tensor, mean, std):
    #         t.mul_(s).add_(m)
    return tensors_


def split_im(img):

    h, w = img.shape[:2]

    w = w // 2

    return img[:, :w], img[:, w:]



def fast_hist(a, b, n=19):
    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)


def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu

def tv_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    # print(dx.size(), dy.size())

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0














