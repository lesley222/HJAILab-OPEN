import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import utils
from networks import *
from config import Config as cfg

class SAC:
    def __init__(self, stn, device=None):
        self.stn = stn
        self.device = device
        self.target_entropy = -cfg.DIM_Z

        self.log_alpha = torch.tensor(0.0).to(device).detach().requires_grad_(True)

        self.actor = Encoder(cfg.STATE_CHANNEL, cfg.DIM_Z).to(device)
        self.decoder = Decoder(cfg.DIM_Z).to(device)

        self.critic1 = Critic(cfg.STATE_CHANNEL, cfg.NF).to(device)
        self.critic2 = Critic(cfg.STATE_CHANNEL, cfg.NF).to(device)

        self.critic1_target = Critic(cfg.STATE_CHANNEL, cfg.NF).to(device)
        self.critic2_target = Critic(cfg.STATE_CHANNEL, cfg.NF).to(device)

        self.eval(self.critic1_target)
        self.eval(self.critic2_target)

        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.LEARNING_RATE/2)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.LEARNING_RATE/2, weight_decay=1e-5)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)

        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def get_value(self, state, action):
        self.critic1.eval()
        self.critic2.eval()
        v1 = self.critic1(state, action)
        v2 = self.critic2(state, action)
        return torch.min(v1, v2).item()

    def choose_action(self, state, test=False):
        self.actor.eval()
        self.decoder.eval()

        dist, enc = self.actor(state)

        if test:
            latent = dist.mean
        else:
            latent = dist.sample()
        field = self.decoder(latent, enc)

        field = field.clamp(-0.5, 0.5)
        # if test:
            # return field.detach()
        return latent.detach(), field.detach()

    def optimize(self, update_actor, update_vae, samples):
        """ update_actor: should update actor now?
            samples: s, a, r, s_, h, h_
        """
        loss = self._loss(samples, update_actor, update_vae)

        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        return loss


    def _loss(self, samples, update_actor, update_vae):
        s, a, r, s_, done = samples

        s = utils.tensor(s, device=self.device)
        a = utils.tensor(a, device=self.device)
        r = utils.tensor(r[..., None], device=self.device)
        s_ = utils.tensor(s_, device=self.device)
        done = utils.tensor(done[..., None],  device=self.device)

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

        critic1_loss = F.mse_loss(Q1, Q_target.detach())
        critic2_loss = F.mse_loss(Q2, Q_target.detach())

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        
        loss['critic1'] = critic1_loss
        loss['critic2'] = critic2_loss

        ########## actor loss  ############            
        if update_actor:
            dist, enc = self.actor(s)

            action = dist.sample()
            log_pi = dist.log_prob(action)
            # alpha loss
            alpha_loss = -(self.log_alpha * (log_pi.unsqueeze(1) + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            loss['alpha'] = alpha_loss

            #############actor loss###############
            actor_Q1 = self.critic1(s, action)
            actor_Q2 = self.critic2(s, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_pi.unsqueeze(1) - actor_Q).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            loss['actor'] = actor_loss

            ########### ncc loss ##########
        if update_vae:
            vae_dist, vae_enc = self.actor(s)
            latent = vae_dist.sample()
            flow = self.decoder(latent, vae_enc)
            warped = self.stn(s[:, 1:], flow)
            ncc_loss = utils.ncc_loss(s[:, :1], warped) + 1.0*utils.gradient_loss(flow)
            
            self.actor_optim.zero_grad()
            self.decoder_optim.zero_grad()
            ncc_loss.backward()
            self.actor_optim.step()
            self.decoder_optim.step()
            
            loss['ncc'] = ncc_loss

        return loss

    @staticmethod
    def soft_update(target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
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

    def load_model(self, name, model_path):
        eval("self.{}.load_state_dict(torch.load('{}'))".format(name, model_path))

    def load_actor(self, model_path):
        self.actor.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    def load_critic(self, critic1_path, critic2_path):
        self.critic1.load_state_dict(torch.load(critic1_path,map_location=torch.device('cpu')))
        self.critic2.load_state_dict(torch.load(critic2_path,map_location=torch.device('cpu')))

    def load_decoder(self, model_path):
        self.decoder.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))




