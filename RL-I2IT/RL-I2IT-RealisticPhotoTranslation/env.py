import torch
import numpy as np
import copy
import torchvision.utils as vutils

from scipy import misc
from random import shuffle
from config import Config as cfg
import utils
import lpips

class Env(object):

    def __init__(self, dataset, device=None):
        self.device = device
        self.generator = self._generator(dataset)
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.reset()


    def _generator(self, datasets, batch_size=1):
        indices = np.arange(len(datasets))
        self.epoch = 0
        while True:
            # np.random.shuffle(indices)
            self.epoch += 1
            # for idx in indices:
                # input = datasets.getitem(idx)
            for input in datasets:

                target = input['A'][0]
                # print(target.shape)
                source = input['B'][0]
                yield target.unsqueeze(0), source.unsqueeze(0)

    def reset(self):
        item = next(self.generator)
        self.target = item[0].to(self.device)
        self.source = item[1].to(self.device)

        # self.moving = torch.zeros_like(self.source).to(self.device)
        # self.moving = torch.clone(self.source)
        self.moving = torch.randn(*self.source.shape).to(self.device)
        self.global_step = 1
        self.prev_score = self.psnr()

        return self.state(), self.target, self.prev_score

    def state(self):
        return torch.cat([self.source, self.moving], dim=1)
        # return self.moving

    def score(self):
        return utils.psnr(
            utils.numpy_im(self.target, self.device),
            utils.numpy_im(self.moving, self.device))
        # return utils.ssim(
        #     utils.numpy_im(self.target, self.device),
        #     utils.numpy_im(self.moving, self.device))
        # return 1. - self.loss_fn(self.target, self.moving).squeeze().item()

    def psnr(self):
        return utils.psnr(
            utils.numpy_im(self.target, self.device),
            utils.numpy_im(self.moving, self.device))

    def done(self):
        return self.score() > cfg.SCORE_THRESHOLD

    def step(self, prediction):
        self.moving = prediction
        score = self.score()

        image_score = self.psnr()
        reward = score

        if self.done():
            reward += 10

        return reward, self.state(), self.done(), image_score

    def save_init(self, dir=cfg.PROCESS_PATH):
        # print(self.source.data.shape)
        vutils.save_image(self.source.data, dir+'/source.png', normalize=True)
        vutils.save_image(self.target.data, dir+'/target.png', normalize=True)

    def save_process(self, idx, dir=cfg.PROCESS_PATH):
        if idx % 2 == 0:
            vutils.save_image(self.moving.data, '{}/moving-{}.png'.format(dir, idx), normalize=True)





























