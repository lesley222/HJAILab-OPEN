import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import os

import utils
from networks import *
from config import Config as cfg

class SAC:
    def __init__(self, stn, device=None):
        self.stn = stn
        self.device = device
        self.target_entropy = -64
        self.scale = 0.001

        self.log_alpha = torch.tensor(0.0).to(device).detach().requires_grad_(True)
        self.criterion = nn.BCELoss().to(device)

        self.actor = Encoder(cfg.STATE_CHANNEL, cfg.NF, cfg.BOTTLE).to(device)
        self.decoder = Decoder(cfg.NF, cfg.BOTTLE).to(device)
        self.netD = NetD(cfg.STATE_CHANNEL // 2, cfg.NF).to(device)
        # self.criterion = nn.BCELoss()

        self.critic1 = Critic(cfg.STATE_CHANNEL, cfg.NF, cfg.BOTTLE).to(device)
        self.critic2 = Critic(cfg.STATE_CHANNEL, cfg.NF, cfg.BOTTLE).to(device)

        self.critic1_target = Critic(cfg.STATE_CHANNEL, cfg.NF, cfg.BOTTLE).to(device)
        self.critic2_target = Critic(cfg.STATE_CHANNEL, cfg.NF, cfg.BOTTLE).to(device)

        self.eval(self.critic1_target)
        self.eval(self.critic2_target)

        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.LEARNING_RATE/10, betas=(0.5, 0.9))
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.netd_optim = torch.optim.Adam(self.netD.parameters(), lr=cfg.LEARNING_RATE/10, betas=(0.5, 0.9))
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))

        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)

        pactor = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        pdecoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        pnetd = sum(p.numel() for p in self.netD.parameters() if p.requires_grad)
        # pcritic1 = sum(p.numel() for p in self.critic1.parameters() if p.requires_grad)
        # pcritic2 = sum(p.numel() for p in self.critic2.parameters() if p.requires_grad)

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
            latent = dist.sample()
        field = self.decoder(latent, enc)

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
        t, s, a, r, s_, done = samples

        t = utils.tensor(t, device=self.device)
        s = utils.tensor(s, device=self.device)
        a = utils.tensor(a, device=self.device)
        r = utils.tensor(r[..., None], device=self.device)
        s_ = utils.tensor(s_, device=self.device)
        done = utils.tensor(done[..., None],  device=self.device)

        real_label = torch.ones((s.size(0), 1), device=self.device)
        fake_label = torch.zeros((s.size(0), 1), device=self.device)

        loss = {}

        ######## critic loss #######
        dist_, enc_ = self.actor(s_)
        a_ = dist_.sample()

        log_pi_next = dist_.log_prob(a_)
        Q_target1_next = self.critic1_target(s_, a_)
        Q_target2_next = self.critic2_target(s_, a_)
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        Q_target = r + (cfg.GAMMA * (1-done) * (Q_target_next - self.alpha * log_pi_next.unsqueeze(1)))

        Q1 = self.critic1(s, a)
        Q2 = self.critic2(s, a)
        # print(r[0].item(), log_pi_next[0].item(), Q_target[0].item(), Q1[0].item())
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
            alpha_loss = -(self.log_alpha * (log_pi.unsqueeze(1) + self.target_entropy).detach()).mean()
            loss['alpha'] = alpha_loss

            self.alpha_optim.zero_grad()
            loss['alpha'].backward()
            self.alpha_optim.step()
            #############actor loss###############

            # mu = torch.zeros(action.shape, device=self.device)
            # logvar = torch.ones(action.shape, device=self.device)
            # policy_prior = self.actor.dist_multivar_normal(mu, logvar)
            # policy_prior_log_probs = policy_prior.log_prob(action)
            # print(policy_prior_log_probs.shape)

            actor_Q1 = self.critic1(s, action)
            actor_Q2 = self.critic2(s, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_pi.unsqueeze(1)*self.scale - actor_Q).mean() #  - policy_prior_log_probs

            loss['actor'] = actor_loss

            # if global_ep >= cfg.FIRST_EP:
            #     actor_fake = self.decoder(action, enc)
            #     actor_fake_D = self.netD(actor_fake)
            #     actor_g_loss = self.criterion(actor_fake_D, real_label)

            #     loss['actor'] += 0.001*actor_g_loss

            self.actor_optim.zero_grad()
            loss['actor'].backward()
            self.actor_optim.step()
        ########### vae loss ##########
        if update_vae:
            vae_dist, vae_enc = self.actor(s)
            latent = vae_dist.sample()
            reconstruction = self.decoder(latent, vae_enc)
            # t = s[:, :3]
            if global_ep % 100 == 0:
                vutils.save_image(t.data, 'process/real.png', normalize=True)
                vutils.save_image(reconstruction.data, 'process/rec.png', normalize=True)
                vutils.save_image(s[:, 3:].data, 'process/state.png', normalize=True)

            ######### l2 loss #########

            size = cfg.HEIGHT

            rec_loss = torch.abs(reconstruction - t).mean()
            tv_loss = utils.tv_loss(reconstruction)

            if global_ep < cfg.FIRST_EP:
                loss['G'] = rec_loss
                loss['vae'] = rec_loss
            else:
                real_D = self.netD(t)
                fake_D = self.netD(reconstruction.detach())

                ############### discriminator loss ###################
                d_rloss = self.criterion(real_D, real_label)
                d_floss = self.criterion(fake_D, fake_label)

                loss['real'] = real_D.mean().item()

                loss['D'] = d_rloss + d_floss

                self.netd_optim.zero_grad()
                loss['D'].backward()
                self.netd_optim.step()
                ############### generator loss ##################
                _fake_D = self.netD(reconstruction)
                g_loss = self.criterion(_fake_D, real_label)

                loss['G'] = 1.*rec_loss + 0.02*g_loss + 0.1*tv_loss
                loss['fake'] = _fake_D.mean().item()
                loss['vae'] = rec_loss

            self.actor_optim.zero_grad()
            self.decoder_optim.zero_grad()
            loss['G'].backward()
            self.actor_optim.step()
            self.decoder_optim.step()
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
        torch.save(self.actor.state_dict(), os.path.join(model_path,'actor_{}.ckpt'.format(step)))
        torch.save(self.critic1.state_dict(), os.path.join(model_path,'critic1_{}.ckpt'.format(step)))
        torch.save(self.critic2.state_dict(), os.path.join(model_path,'critic2_{}.ckpt'.format(step)))
        torch.save(self.decoder.state_dict(), os.path.join(cfg.MODEL_PATH,'decoder_{}.ckpt'.format(step)))
        torch.save(self.netD.state_dict(), os.path.join(cfg.MODEL_PATH,'netD_{}.ckpt'.format(step)))

    def load_model(self, name, model_path):
        eval("self.{}.load_state_dict(torch.load('{}'))".format(name, model_path))

    def load_actor(self, model_path):
        self.actor.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    def load_critic(self, critic1_path, critic2_path):
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def load_decoder(self, model_path):
        self.decoder.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    def load_netD(self, model_path):
        self.netD.load_state_dict(torch.load(model_path))




