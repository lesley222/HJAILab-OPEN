import math
import os
import random
import cv2
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
# from imagehash import phash
# from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim
# from sklearn.metrics.pairwise import cosine_similarity


def setup_seed(seed):
    """
    设置随机数种子
    :param seed: int
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # cpu\gpu 结果一致


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


# def compute_ssim(data):
#     std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
#     mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
#     img = data.clone().numpy()
#     img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
#     return img

# G = FF^T
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


# using ImageNet values
def normalize_tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def get_path(root, paths):
    new_paths = []
    for path in paths:
        new_paths.append(os.path.join(root, path))
    return new_paths


def numpy_im(data, device=None):
    im = (numpy(data, device).squeeze().transpose(1,2,0)+1) * 127.5
    return im


def numpy(data, device=None):
    if device is not None:
        return data.detach().cpu().numpy()
    else:
        return data.detach().numpy()


def to_numpy(data, device=None):
    if device is not None:
        return data.detach().cpu().numpy()
    else:
        return data.detach().numpy()


def to_numpy_img(data, device=None):
    im = (to_numpy(data, device).squeeze().transpose(1, 2, 0) + 1) * 127.5
    return im  # im.shape: (128, 128, 3)


def gram_score(style_maps, result_maps):
    for i in range(len(style_maps)):
        style_G = gram(style_maps[i].fea.cpu().detach())
        result_G = gram(result_maps[i].fea.cpu().detach())
        print(torch.sum(style_G - result_G))


def save_data(ssim, content_loss, style_loss, time, name=None):
    # data = pd.DataFrame([np.array(ssim), np.array(content_loss), np.array(style_loss), np.array(time)],
    #                     columns=['ssim', 'content_loss', 'style_loss', 'time'])
    data = pd.DataFrame({'ssim':ssim,
                         'content_loss':content_loss,
                         'style_loss':style_loss,
                         'time':time})
    data.to_csv("./test_data_logs/"+name+".csv")


# from math import exp
# import torch.nn.functional as F
# def ssim_self(image1, image2, K, window_size, L):
#     _, channel1, _, _ = image1.size()
#     _, channel2, _, _ = image2.size()
#     channel = min(channel1, channel2)
#
#     # gaussian window generation
#     sigma = 1.5  # default
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     _1D_window = (gauss / gauss.sum()).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#
#     # define constants
#     # * L = 255 for constants doesn't produce meaningful results; thus L = 1
#     # C1 = (K[0]*L)**2;
#     # C2 = (K[1]*L)**2;
#     C1 = K[0] ** 2
#     C2 = K[1] ** 2
#
#     mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     return ssim_map.mean()
