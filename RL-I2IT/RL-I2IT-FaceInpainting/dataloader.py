import torch
import numpy as np
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.datasets as dset
import cv2
import os
import glob

import utils

class Celeba:
    def __init__(self, img_size=128, hole_size=32, mode='train', path='../datasets'):
        super().__init__()
        self.hole_size = hole_size
        self.mode = mode
        self.path = path
        self.transform = transform(img_size)
        self.names = self._load_data(mode, path)
        print('the {} data is {}'.format(mode, len(self.names)))

    def _load_data(self, mode, path):
        with open(os.path.join(path, 'list_eval_partition.txt'), 'r') as f:
            lines = f.readlines()
        partitions = []
        for line in lines:
            name, num = line.strip('\n').split(' ')
            if mode == 'train' and num == '0' \
                or mode == 'eval' and num == '1' \
                or mode == 'test' and num == '2':
                partitions.append(os.path.join(path,'celeba', name))
        return partitions

    def __len__(self):
        return len(self.names)

    def getitem(self, idx):
        name = self.names[idx]
        # print(name)
        img = Image.open(name)
        img = self.transform(img)
        img_cut, mask, x, y = utils.cut_im(img, self.hole_size)

        return img_cut, img, mask, x, y

    def __getitem__(self, idx):
        return self.getitem(idx)

    def generator(self):
        idx = 0
        while idx < len(self.names):
            yield self.getitem(idx)
            idx += 1


class CelebaHQ:
    def __init__(self, img_size=256, hole_size=128, mode='train', path='../datasets/celeba-hq'):
        super().__init__()
        self.img_size = img_size
        self.hole_size = hole_size
        self.mode = mode
        self.path = path
        self.transform = transform(img_size)
        self.names = self._load_data(mode, path)
        print('the {} data is {}'.format(mode, len(self.names)))

    def _load_data(self, mode, path):
        root = os.path.join(path, 'celeba-'+str(self.img_size))
        path_pattern = os.path.join(root, '*.jpg')
        names = sorted(glob.glob(path_pattern))
        partitions = []
        for i, name in enumerate(names):
            # print(name)
            if mode == 'train' and i < 28000 \
                or mode == 'test' and i>=28000:
                partitions.append(name)
        return partitions

    def __len__(self):
        return len(self.names)

    def getitem(self, idx):
        name = self.names[idx]
        # print(name)
        img = Image.open(name)
        img = self.transform(img)
        img_cut, mask, x, y = utils.cut_im(img, self.hole_size)

        return img_cut, img, mask, x, y

    def __getitem__(self, idx):
        return self.getitem(idx)

    def generator(self):
        idx = 0
        while idx < len(self.names):
            yield self.getitem(idx)
            idx += 1


def transform(size):
    transforms = []
    transforms.append(T.Resize((size, size)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(transforms)

