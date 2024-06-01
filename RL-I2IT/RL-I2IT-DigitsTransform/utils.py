import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import shutil
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

def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


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
    return im.convert('L')

def resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def read_data(root):
    ct_list = os.listdir(root+'/CT')
    mr_list = os.listdir(root+'/MR')

    ct_list.sort()
    mr_list.sort()

    datas = [(root+'/CT/'+ct, root+'/MR/'+mr) for ct, mr in zip(ct_list, mr_list)]
    return datas


def tensor_im(data):
    im = Image.fromarray(data).convert('L')
    return TF.to_tensor(im)

def tensor(data, dtype=torch.float32, device=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def numpy_im(data, scale=255., device=None):
    im = numpy(data, device).squeeze() * scale
    return im.astype(np.uint8)

def numpy(data, device=None):
    if device is not None:
        return data.detach().cpu().numpy()
    else:
        return data.detach().numpy()

def render_flow(flow, coef = 5, thresh = 250):
    im_flow = np.zeros((3, cfg.HEIGHT, cfg.WIDTH))
    im_flow[1:] = flow
    im_flow = im_flow.transpose(1, 2, 0)
    #im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    #im_flow = 1 - im_flow / 20
    return im_flow


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def ncc_loss(fixed_patch, moving_patch):
    return -ncc_tensor_score(fixed_patch, moving_patch)

def ncc_tensor_score(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J

    sum_filt = torch.ones([1, 1, *win])

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    # print(dx.size(), dy.size())

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

def dice_loss(pred, target):
    return -dice_score(pred, target)

def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 *  torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-10
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice


def random_rotation(im, angle=0):
    image = Image.fromarray(im, mode='L')
    degrees = np.random.uniform(-angle, angle)
    image = TF.rotate(image, degrees)
    return np.array(image)

def rotate(image, angle, scale=[0.3, 1.7]):
    h, w = image.shape

    center = (w / 2, h / 2)
    degrees = np.random.uniform(-angle, angle)
    scales = np.random.uniform(scale[0], scale[1])
    m = cv2.getRotationMatrix2D(center, degrees, scales)
    return cv2.warpAffine(image, m, (w, h))



