import torch
import SimpleITK as sitk
import numpy as np
import copy
import torchvision.utils as vutils

from scipy import misc
from random import shuffle
from config import Config as cfg
import utils


class Env(object):

    def __init__(self, dataset, device=None):
        self.device = device
        self.generator = self._generator(dataset)
        self.reset()

    def _generator(self, datasets, batch_size=1):
        indices = np.arange(len(datasets))
        self.epoch = 0
        while True:
            np.random.shuffle(indices)
            self.epoch += 1
            cut_ims = []
            ims = []
            masks = []
            for idx in indices:
                item = datasets.getitem(idx)
                cut_ims.append(item[0])
                ims.append(item[1])
                masks.append(item[2])
                if len(ims) == batch_size:
                    yield torch.stack(cut_ims, dim=0), torch.stack(ims, dim=0), torch.stack(masks, dim=0)
                    cut_ims, ims, masks = [], [], []

    def reset(self):
        item = next(self.generator)
        self.cut_im = item[0].to(self.device)
        self.im = item[1].to(self.device)

        self.mask = item[2].to(self.device)

        self._state = self.cut_im.clone()
        self.field = utils.get_area(self.cut_im, cfg.HOLE_SIZE)

        self.global_step = 1
        self.prev_score = self.score()

        return self.state(), self.im, self.mask, self.prev_score

    def state(self):
        return utils.merge_area(self.cut_im, self.field)

    def score(self):
        real_mask = utils.get_area(self.im, cfg.HOLE_SIZE)
        return utils.psnr(utils.numpy_im(self.field, self.device), utils.numpy_im(real_mask, self.device))
        # return -(self.field - real_mask).abs().mean().item()

    def done(self):
        return self.score() > cfg.SCORE_THRESHOLD

    def step(self, field):
        self.field = field
        # self.field = self.field.clamp(-1, 1)
        self._state = self.state()
        image_score = self.score()

        reward = image_score
        # self.prev_score = image_score

        if self.done():
            reward += 10

        return reward, self.state(), self.done(), image_score

    def save_init(self, dir=cfg.PROCESS_PATH):
        real_mask = utils.get_area(self.im, cfg.HOLE_SIZE)
        vutils.save_image(real_mask.data, dir+'/real_mask.png', normalize=True)
        vutils.save_image(self.im.data, dir+'/im.png', normalize=True)
        vutils.save_image(self.cut_im.data, dir+'/im_cut.png', normalize=True)

    def save_process(self, idx, dir=cfg.PROCESS_PATH):
        if idx % 2 == 0:
            # vutils.save_image(self.cut_im[0].data, dir+'/im_cut_.png', normalize=True)
            vutils.save_image(self._state.data, '{}/image-{}-moved.png'.format(dir, idx), normalize=True)
            vutils.save_image(self.field.data, '{}/field.png'.format(dir, idx), normalize=True)





























