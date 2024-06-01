import torch
import numpy as np
from mnist import MNIST
import cv2

import utils

class MnistData:
    def __init__(self, root, mode='train'):
        np.random.seed(100)
        self.mndata = MNIST(root)
        if mode == 'train':
            self.images, self.labels = self.mndata.load_training()
        else:
            self.images, self.labels = self.mndata.load_testing()
        self.data_classified = self.load_data()

    def load_data(self):
        datas = [[] for _ in range(10)]
        for image, label in zip(self.images, self.labels):
                datas[label].append(image)

        for i in range(len(datas)):
            print(i, ':', len(datas[i]))
        return datas

    def __len__(self):
        return len(self.labels)

    def next_same_class(self):
        label = np.random.randint(len(self.data_classified))
        moving_idx = np.random.randint(len(self.data_classified[label]))
        fixed_idx = np.random.randint(len(self.data_classified[label]))

        moving = self.data_classified[label][moving_idx]
        fixed = self.data_classified[label][fixed_idx]

        return (np.reshape(moving, (28, 28)).astype(np.float32),
                    np.reshape(fixed, (28, 28)).astype(np.float32))


    def next_diff_class(self, fixed_label=None, moving_label=None, random_rotation=None):
        if fixed_label is None:
            fixed_label = np.random.randint(len(self.data_classified))
        fixed_idx = np.random.randint(len(self.data_classified[fixed_label]))

        if moving_label is None:
            moving_label = np.random.randint(len(self.data_classified))
        moving_idx = np.random.randint(len(self.data_classified[moving_label]))

        moving = np.reshape(self.data_classified[moving_label][moving_idx], (28, 28)).astype(np.float32)
        fixed = np.reshape(self.data_classified[fixed_label][fixed_idx], (28, 28)).astype(np.float32)

        if random_rotation:
            moving = utils.rotate(moving, 360)
            fixed = utils.rotate(fixed, 360)

        return moving, fixed


    def same_data(self, fixed_label, moving_label, label_idx):
        moving = np.reshape(self.data_classified[moving_label][label_idx], (28, 28)).astype(np.float32)
        fixed = np.reshape(self.data_classified[fixed_label][label_idx], (28, 28)).astype(np.float32)

        return moving, fixed

    def read_test(self, idx):
        moving = cv2.imread('test/{}-moving.bmp'.format(idx), cv2.IMREAD_GRAYSCALE)
        fixed = cv2.imread('test/{}-fixed.bmp'.format(idx), cv2.IMREAD_GRAYSCALE)

        return moving, fixed




