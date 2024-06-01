import torch
import numpy as np
import cv2
import copy
from random import shuffle
from config import Config as cfg
from networks import SpatialTransformer
import utils


class Env(object):
    """
    The environment must specify a root directory including paired files in CT and MR respectively
    The virtual_label is used to assist the registration learning process
    """
    def __init__(self, mnist, stn, device=None):
        self.stn = stn
        self.device = device
        self.mnist = mnist
        self.reset()

    def reset(self):
        self.moving, self.fixed = self.mnist.next_diff_class()
        self.moved = copy.deepcopy(self.moving)

        self.field = None
        self.prev_score = self.image_dice()
        return self.state(), self.prev_score

    def state(self):
        # print(self.moved)
        moved = utils.tensor_im(self.moved)
        fixed = utils.tensor_im(self.fixed)
        return torch.cat([fixed, moved], dim=0).unsqueeze(0).to(self.device)

    def dice(self, im1, im2):
        return utils.dice(im1>0, im2>0)[0]

    def image_dice(self):
        return self.dice(self.moved, self.fixed)

    def done(self):
        return self.image_dice() > cfg.SCORE_THRESHOLD

    # latent is a tensor representing the displacement field
    def step(self, field):
        if self.field is None:
            self.field = field
        else:
            self.field = self.stn(self.field, field) + field

        predict_field = self.field

        warped = self.stn(utils.tensor_im(self.moving).unsqueeze(0).to(self.device), predict_field)
        self.moved = utils.numpy_im(warped, device=self.device)

        image_score = self.image_dice()
        reward = image_score * 10

        return reward, self.state(), self.done(), image_score


    def save_init(self, dir=cfg.PROCESS_PATH):
        cv2.imwrite(dir+'/moving.bmp', self.moving)
        cv2.imwrite(dir+'/fixed.bmp', self.fixed)

    def save_process(self, idx, dir=cfg.PROCESS_PATH):
        if idx % 2 == 0:
            cv2.imwrite('{}/image-{}-moved.bmp'.format(dir, idx), self.moved)




























