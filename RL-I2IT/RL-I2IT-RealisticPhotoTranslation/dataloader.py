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

class Dataset:
    def __init__(self, img_size=128, mode='train', path='../datasets/facades/train'):
        self.mode = mode
        self.path = path
        self.img_size = img_size
        self.transform_source = transform(img_size, norm=False)
        self.transform_target = transform(img_size, norm=True)
        self.names = self._load_data(path)
        print('the {} data is {}'.format(mode, len(self.names)))

    def _load_data(self, path):
        print(path)
        names = list(glob.glob(path+'/*.jpg'))
        names.sort()
        return names

    def __len__(self):
        return len(self.names)

    def getitem(self, idx):
        name = self.names[idx]
        img = utils.load_image(name)
        target = img.crop((0, 0, self.img_size, self.img_size))
        source = img.crop((self.img_size, 0, self.img_size*2, self.img_size))

        if np.random.uniform() > 0.5 and self.mode == 'train':
            x = np.random.randint(self.img_size//2)
            y = np.random.randint(self.img_size//2)
            flip = np.random.uniform() > 0.5

            target = target.crop((x, y, self.img_size, self.img_size))
            source = source.crop((x, y, self.img_size, self.img_size))

            if flip:
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
                source = source.transpose(Image.FLIP_LEFT_RIGHT)

        source = self.transform_source(source)
        target = self.transform_target(target)
        return target, source

    def __getitem__(self, idx):
        return self.getitem(idx)

    def generator(self):
        idx = 0
        while idx < len(self.names):
            yield self.getitem(idx)
            idx += 1

def transform(size, norm=False):
    transforms = []
    transforms.append(T.Resize((size+30)))
    transforms.append(T.CenterCrop((size)))
    transforms.append(T.ToTensor())
    if norm:
        transforms.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(transforms)

