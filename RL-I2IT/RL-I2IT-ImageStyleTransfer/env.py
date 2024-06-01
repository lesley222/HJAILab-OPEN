import torch
import torch.nn.functional as F
import numpy as np
import copy
import torchvision.utils as vutils
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim
from vgg import Vgg16
from scipy import misc
from random import shuffle
from config import Config as cfg
import utils


class Env(object):

    def __init__(self, dataloader, style, device=None, max_episode_length=None):
        self.device = device  # CPU or GPU
        self.generator = self._generator(dataloader)
        self.vgg = Vgg16().to(device)
        self.style = style.to(device)
        self.loss_mse = torch.nn.MSELoss()
        self.reset()

    def _generator(self, dataloader):
        self.epoch = 0
        while True:
            self.epoch += 1
            for content in dataloader:
                yield content

    def reset(self):
        self.global_step = 1
        content = next(self.generator)
        self.content = content.to(self.device)  # [n, 3, 256, 256]
        self.moving = torch.clone(self.content)
        # self.moving = torch.randn(*self.content.shape).to(self.device)

        return self.content, self.style, self.state()

    def state(self):
        # return torch.cat([self.content, self.moving], dim=1)
        return self.moving

    def score(self):
        y_hat_features = self.vgg(self.moving)

        style_features = self.vgg(self.style)
        style_gram = [utils.gram(fmap) for fmap in style_features]

        y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]  # gram
        style_loss = 0.0
        for j in range(4):
            style_loss += self.loss_mse(y_hat_gram[j], style_gram[j][:1])
        style_loss = cfg.STYLE_WEIGHT * style_loss * 10

        return -style_loss.cpu().detach().item()

    def done(self):
        return self.score() > cfg.SCORE_THRESHOLD

    def step(self, prediction):
        self.moving = prediction
        reward = self.score()

        image_score = self.score()  # summary

        if self.done():
            reward += 1.0

        return reward, self.state(), self.done(), image_score
























