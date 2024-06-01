import os

import torch
import torchvision.utils as vutils
from vgg import Vgg16

import utils


class Env(object):

    def __init__(self, dataloader, style, args, device=None):
        self.max_episode_steps = args.max_episode_steps
        self.args = args
        self.device = device
        self.generator = self._generator(dataloader)
        self.vgg = Vgg16().to(device).eval()
        self.loss_fn = torch.nn.MSELoss()
        self.style = style.to(device)
        self.reset()

    def _generator(self, dataloader):
        self.epoch = 0
        while True:
            self.epoch += 1
            for frame, video_name in dataloader:
                yield frame, video_name

    def reset(self):
        frame, video_name = next(self.generator)
        self.frame = frame.to(self.device)
        self.video_name = video_name

        self.moving = torch.clone(self.frame)
        self.global_step = 1
        self.prev_score = self.score()

        return self.state(), self.prev_score, video_name

    def state(self):
        return self.moving

    def score(self):
        y_hat_features = self.vgg(self.moving)

        style_features = self.vgg(self.style)
        style_gram = [utils.gram(fmap) for fmap in style_features]

        y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]  # gram
        style_loss = 0.0
        for j in range(4):
            style_loss += self.loss_fn(y_hat_gram[j], style_gram[j][:1])
        style_loss = self.args.style_weight * style_loss

        return -style_loss.cpu().detach().item()

    def done(self):
        return self.score() > self.args.score_threshold

    def step(self, prediction):
        self.moving = prediction
        score = self.score()

        image_score = self.score()
        reward = score

        if self.done():
            reward += 10

        return self.state(), reward, self.done(), image_score

    def save_init(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        vutils.save_image(self.frame.data, dir + '/content.png', normalize=True)
        vutils.save_image(self.style.data, dir + '/style.png', normalize=True)

    def save_process(self, idx, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        utils.save_image('{}/moving-{}.png'.format(dir, idx), self.moving.cpu().data[0])
