import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import os

import utils
from network import *
from vgg import Vgg16
from config import Config as cfg


class SAC:
    def __init__(self, device=None):
        self.device = device
        self.target_entropy = -64
        self.scale = 0.1

        self.log_alpha = torch.tensor(-0.3).to(device).detach().requires_grad_(True)

        self.criterion = nn.BCELoss().to(device)

        self.actor = Encoder().to(device)
        self.decoder = Decoder().to(device)

        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)

        self.critic1_target = Critic().to(device)
        self.critic2_target = Critic().to(device)

        self.eval(self.critic1_target)
        self.eval(self.critic2_target)

        self.vgg = Vgg16().to(device)

        self.loss_mse = torch.nn.MSELoss()

        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.LEARNING_RATE / 10, betas=(0.5, 0.9))
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.LEARNING_RATE / 10, betas=(0.5, 0.9))

        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)

        pactor = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        pdecoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        print('total parameters: {}'.format(pactor + pdecoder))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def get_value(self, state, action):
        self.critic1.eval()
        self.critic2.eval()
        v1 = self.critic1(state, action)
        v2 = self.critic2(state, action)
        return torch.min(v1, v2).mean().item()

    def choose_action(self, state, test=False):
        self.actor.eval()
        self.decoder.eval()

        dist, enc = self.actor(state)

        if test:
            latent = dist.mean
        else:
            latent = dist.sample()  # latent.size() [1, 256]
        field = self.decoder(latent, enc)  # [1, 3, 128, 128]

        return latent.detach(), field.detach()

    def optimize(self, global_ep, update_actor, update_vae, samples):
        """ update_actor: should update actor now?
            samples: s, a, r, s_, h, h_
        """
        self.actor.train()
        self.decoder.train()
        self.critic1.train()
        self.critic2.train()

        loss = self._loss(global_ep, samples, update_actor, update_vae)

        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        return loss

    def _loss(self, global_ep, samples, update_actor, update_vae):
        c, t, s, a, r, s_, done = samples  # target, state, action, reward, next_state, done

        c = torch.as_tensor(c, dtype=torch.float32, device=self.device)  # [n, 3, 128, 128]
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)  # [n, 3, 128, 128]
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)  # [n, 6, 128, 128]
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)  # [n, 256]
        r = torch.as_tensor(r[..., None], dtype=torch.float32, device=self.device)  # [n, 1]
        s_ = torch.as_tensor(s_, dtype=torch.float32, device=self.device)  # [n, 6, 128, 128]
        done = torch.as_tensor(done[..., None], dtype=torch.float32, device=self.device)  # [n, 1]

        loss = {}

        ######## critic loss #######
        dist_, enc_ = self.actor(s_)
        a_ = dist_.sample()  # [1,1,64,64]
        log_pi_next = dist_.log_prob(a_)

        Q_target1_next = self.critic1_target(s_, a_)
        Q_target2_next = self.critic2_target(s_, a_)
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        Q_target = r + (cfg.GAMMA * (1 - done) * (Q_target_next - self.alpha * log_pi_next))
        Q1 = self.critic1(s, a)  # [1, 1]
        Q2 = self.critic2(s, a)

        critic1_loss = F.mse_loss(Q1, Q_target.detach())
        critic2_loss = F.mse_loss(Q2, Q_target.detach())

        loss['critic1'] = critic1_loss
        loss['critic2'] = critic2_loss

        self.critic1_optim.zero_grad()
        loss['critic1'].backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        loss['critic2'].backward()
        self.critic2_optim.step()

        ########## actor loss  ############
        if update_actor:
            dist, enc = self.actor(s)
            action = dist.sample()
            log_pi = dist.log_prob(action)

            # alpha loss
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            loss['log_pi'] = log_pi.mean()
            loss['alpha_loss'] = alpha_loss
            loss['alpha'] = self.alpha

            self.alpha_optim.zero_grad()
            loss['alpha'].backward()
            self.alpha_optim.step()

            ############# actor loss ###############
            actor_Q1 = self.critic1(s, action)
            actor_Q2 = self.critic2(s, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            loss['actor_Q'] = actor_Q.mean()
            actor_loss = (self.alpha.detach() * log_pi * self.scale - actor_Q).mean()  # - policy_prior_log_probs

            loss['actor'] = actor_loss
            # print('actor_loss:', actor_loss)

            self.actor_optim.zero_grad()
            loss['actor'].backward()
            self.actor_optim.step()

        ########### vae loss ##########
        if update_vae:
            vae_dist, vae_enc = self.actor(s)
            latent = vae_dist.sample()
            reconstruction = self.decoder(latent, vae_enc)

            # get vgg feature
            y_c_features = self.vgg(c)
            y_hat_features = self.vgg(reconstruction)

            style_features = self.vgg(t)
            style_gram = [utils.gram(fmap) for fmap in style_features]

            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += self.loss_mse(y_hat_gram[j], style_gram[j][:1])
            style_loss = cfg.STYLE_WEIGHT * style_loss

            # (h_relu_2_2)
            recon = y_c_features[1]
            recon_hat = y_hat_features[1]
            content_loss = cfg.CONTENT_WEIGHT * self.loss_mse(recon_hat, recon)

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]))
            tv_loss = cfg.TV_WEIGHT * (diff_i + diff_j)

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            self.actor_optim.zero_grad()
            self.decoder_optim.zero_grad()
            total_loss.backward()
            self.actor_optim.step()
            self.decoder_optim.step()

            loss['total_loss'] = total_loss
            loss['content_loss'] = content_loss
            loss['style_loss'] = style_loss
            loss['tv_loss'] = tv_loss

        return loss

    @staticmethod
    def soft_update(target, source, tau=0.005):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def eval(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def save_model(self, step, model_path):
        self.actor.eval()
        self.decoder.eval()
        self.critic1.eval()
        self.critic2.eval()
        torch.save(self.actor.state_dict(), os.path.join(model_path, 'actor_' + cfg.NAME + '_{}.ckpt'.format(step)))
        torch.save(self.critic1.state_dict(), os.path.join(model_path, 'critic1_' + cfg.NAME + '_{}.ckpt'.format(step)))
        torch.save(self.critic2.state_dict(), os.path.join(model_path, 'critic2_' + cfg.NAME + '_{}.ckpt'.format(step)))
        torch.save(self.decoder.state_dict(), os.path.join(model_path, 'decoder_' + cfg.NAME + '_{}.ckpt'.format(step)))
        self.actor.train()
        self.decoder.train()
        self.critic1.train()
        self.critic2.train()

    def load_model(self, name, model_path):
        eval("self.{}.load_state_dict(torch.load('{}'))".format(name, model_path))

    def load_actor(self, model_path):
        print('Loading actor: ', model_path)
        self.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def load_critic(self, critic1_path, critic2_path):
        print('Loading critic1: ', critic1_path)
        self.critic1.load_state_dict(torch.load(critic1_path))
        print('Loading critic2: ', critic2_path)
        self.critic2.load_state_dict(torch.load(critic2_path))

    def load_decoder(self, model_path):
        print('Loading decoder: ', model_path)
        self.decoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
