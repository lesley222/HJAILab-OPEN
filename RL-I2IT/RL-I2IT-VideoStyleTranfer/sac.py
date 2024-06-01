import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, Executor, Echo
from vgg import Vgg16
from temporalLoss import TemporalLoss
import utils


class SAC(object):
    def __init__(self, style, args, device):

        self.device = device

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.args = args

        self.policy_type = args.policy
        self.actor_update_interval = args.actor_update_interval
        self.vae_update_interval = args.vae_update_interval
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.style = style.to(self.device)
        self.vgg = Vgg16().to(self.device).eval()
        self.loss_fn = nn.MSELoss()
        self.temporal_loss = TemporalLoss(device=self.device)

        self.executor = Executor(self.args.action_dim, self.args.hidden_dim).to(self.device)
        self.exe_optim = Adam(self.executor.parameters(), lr=args.lr)

        self.echo = Echo(self.args.action_dim).to(self.device)
        self.echo_optim = Adam(self.echo.parameters(), lr=args.lr)

        self.critic = QNetwork(self.args.state_input_dim, self.args.action_dim, self.args.hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(self.args.state_input_dim, self.args.action_dim, self.args.hidden_dim).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor((64, 64, 64)).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(self.args.state_input_dim, self.args.action_dim, self.args.hidden_dim).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.args.state_input_dim, self.args.action_dim, self.args.hidden_dim).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        pactor = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        pexecutor = sum(p.numel() for p in self.executor.parameters() if p.requires_grad)
        print('total parameters: {}'.format(pactor + pexecutor))

    def select_action(self, state, step, evaluate=False):
        # state = torch.FloatTensor(state).to(self.device)
        if evaluate is False:
            action, _, _, enc = self.policy.sample(state)
        else:
            _, _, action, enc = self.policy.sample(state)
        action = self.echo(action, step)
        rec = self.executor(action, enc)
        return action.detach(), rec.detach()

    def update_parameters(self, memory, batch_size, updates, writer):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # --------------------------- update critic ---------------------------------
        if self.args.critic_update:
            with torch.no_grad():
                next_state_action, next_state_log_pi, _, enc = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                next_state_log_pi = next_state_log_pi.view(next_state_log_pi.size(0), -1)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.mean(dim=1)
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch,
                                   action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()
            writer.add_scalar('loss/critic_1', qf1_loss.item(), updates)
            writer.add_scalar('loss/critic_2', qf2_loss.item(), updates)

        # -------------------------- update actor ------------------------------
        if self.args.actor_update:
            if updates % self.actor_update_interval == 0:
                pi, log_pi, _, enc = self.policy.sample(state_batch.detach())

                qf1_pi, qf2_pi = self.critic(state_batch.detach(), pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                log_pi = log_pi.view(log_pi.size(0), -1).mean(dim=1)
                policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                writer.add_scalar('loss/policy', policy_loss.item(), updates)

        # ------------------------- update alpha -----------------------------------
        if self.automatic_entropy_tuning:
            pi, log_pi, _, enc = self.policy.sample(state_batch)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        writer.add_scalar('loss/entropy_loss', alpha_loss.item(), updates)
        writer.add_scalar('entropy_temperature/alpha', alpha_tlogs.item(), updates)

        # -------------------------- update target critic -----------------------------------
        if self.args.critic_update:
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

        # -------------------------- update vae ---------------------------------
        if self.args.vae_update:
            if updates % self.vae_update_interval == 0:
                latent, _, _, enc = self.policy.sample(state_batch)
                rec = self.executor(latent, enc)

                # temporal loss
                second_frame, forward_flow = self.temporal_loss.GenerateFakeData(state_batch)
                second_frame_action, _, _, enc_ = self.policy.sample(second_frame)
                styled_second_frame = self.executor(second_frame_action, enc_)

                temporal_loss = self.temporal_loss(rec, styled_second_frame, forward_flow) * self.args.temp_weight

                # get vgg feature
                y_c_features = self.vgg(state_batch)
                y_hat_features = self.vgg(rec)

                style_features = self.vgg(self.style)
                style_gram = [utils.gram(fmap) for fmap in style_features]

                y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
                style_loss = 0.0
                for j in range(4):
                    style_loss += self.loss_fn(y_hat_gram[j], style_gram[j][:1])
                style_loss = self.args.style_weight * style_loss

                # (h_relu_2_2)
                recon = y_c_features[1]
                recon_hat = y_hat_features[1]
                content_loss = self.args.content_weight * self.loss_fn(recon_hat, recon)

                # calculate total variation regularization (anisotropic version)
                # https://www.wikiwand.com/en/Total_variation_denoising
                diff_i = torch.sum(torch.abs(rec[:, :, :, 1:] - rec[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(rec[:, :, 1:, :] - rec[:, :, :-1, :]))
                tv_loss = self.args.tv_weight * (diff_i + diff_j)

                # total loss
                vae_loss = style_loss + content_loss + tv_loss + temporal_loss
                # vae_loss = vae_loss.detach_().requires_grad_(True)

                self.policy_optim.zero_grad()
                self.exe_optim.zero_grad()
                self.echo_optim.zero_grad()
                vae_loss.backward()
                self.echo_optim.step()
                self.policy_optim.step()
                self.exe_optim.step()
                print('count:{}, total loss:{}, content loss:{}, style loss:{}, tv loss:{}, temporal loss:{}'.format(
                    updates, vae_loss, content_loss, style_loss, tv_loss, temporal_loss))
                writer.add_scalar('loss/vae', vae_loss.item(), updates)

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_1{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'executor_state_dict': self.executor.state_dict(),
                    'echo_state_dict': self.echo.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'executor_optimizer_state_dict': self.exe_optim.state_dict(),
                    'echo_optimizer_state_dict': self.echo_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.executor.load_state_dict(checkpoint['executor_state_dict'])
            self.echo.load_state_dict(checkpoint['echo_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.exe_optim.load_state_dict(checkpoint['executor_optimizer_state_dict'])
            self.echo_optim.load_state_dict(checkpoint['echo_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.executor.eval()
                self.echo.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.executor.train()
                self.echo.train()
                self.critic.train()
                self.critic_target.train()
