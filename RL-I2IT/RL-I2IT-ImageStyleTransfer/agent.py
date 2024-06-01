import torch
from torchvision import datasets, transforms
import numpy as np
# import h5py
import time
import os
from collections import namedtuple

import utils
from config import Config as cfg
from memory import ReplayMemory

Memory = namedtuple('Memory', ['content', 'style', 'state', 'action', 'reward', 'next_state', 'done'])


class Agent:
    def __init__(self, brain, env, summary, device=None):
        super(Agent, self).__init__()
        self.device = device
        self.brain = brain  # SAC
        self.env = env
        self.writer = summary

        self.memories = ReplayMemory(cfg.MEMORY_SIZE, look_forward_steps=0)

    def feed_memory(self, c, t, s, a, r, s_, d):
        memory = Memory(c, t, s, a, r, s_, d)  # content, target, state, action, reward, next_state, done
        self.memories.store(memory)

    def run(self, seed):
        utils.setup_seed(seed)
        tic = time.time()
        print('start training!')

        total_step = 1
        global_episode = 0

        rewards = []
        # ------------------------------------- Episodes -------------------------------------
        while global_episode < cfg.MAX_GLOBAL_EPISODES:
            episode_reward = 0
            steps = 0
            episode_values = []
            scores = []

            content, style, state = self.env.reset()  # content, style, state(content)
            # ---------------------------------- episode ---------------------------------------
            while True:
                latent, field = self.brain.choose_action(state)
                value = self.brain.get_value(state, latent)
                r, s_, done, score = self.env.step(field)
                self.writer.add_scalar(tag='reward', scalar_value=r, global_step=total_step)
                scores.append(score)

                episode_reward = r if steps == 0 else (episode_reward * 0.99 + r * 0.01)
                episode_values.append(value)

                if done:
                    print("No.Episode: ", global_episode,
                          "\tEp_Rewards = ", episode_reward,
                          "\tEp_Steps = ", steps,
                          '\tDone_score = ', score,
                          '\tDone!')

                if steps % 3 == 0:
                    self.feed_memory(utils.to_numpy(content, device=self.device),
                                     utils.to_numpy(style, device=self.device),
                                     utils.to_numpy(state, device=self.device),
                                     utils.to_numpy(latent, device=self.device),
                                     r,
                                     utils.to_numpy(s_, device=self.device),
                                     done)

                if total_step % (cfg.FREQUENCY_ACTOR * cfg.UPDATE_GLOBAL_ITER) == 0:
                    update_actor = True
                else:
                    update_actor = False

                if total_step % (cfg.FREQUENCY_VAE * cfg.UPDATE_GLOBAL_ITER) == 0:
                    update_vae = True
                else:
                    update_vae = False

                if global_episode >= cfg.EP_BOOTSTRAP and total_step % cfg.UPDATE_GLOBAL_ITER == 0:
                    if len(self.memories) < cfg.SAMPLE_BATCH_SIZE:
                        samples = self.memories.sample(len(self.memories))
                    else:
                        samples = self.memories.sample(cfg.SAMPLE_BATCH_SIZE)

                    loss = self.brain.optimize(global_episode, update_actor, update_vae, samples)

                    if update_actor:
                        critic = loss['critic1'].item(), loss['critic2'].item()
                        actor = loss['actor'].item()
                        alpha_loss = loss['alpha_loss']
                        alpha = loss['alpha'].item()
                        log_pi = loss['log_pi'].item()
                        actor_Q = loss['actor_Q'].item()
                        self.writer.add_scalar(tag='log_pi', scalar_value=float(log_pi), global_step=total_step)
                        self.writer.add_scalar(tag='actor_Q', scalar_value=float(actor_Q), global_step=total_step)

                        G = loss['total_loss']
                        content_loss = loss['content_loss']
                        style_loss = loss['style_loss']
                        print('episode-{}-{}:'.format(global_episode, steps),
                              ' ----- critic1, critic2, actor, vae, content, style, score: ',
                              '{:.2f}, {:.4f}, {:.4f},  {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                                critic[0], critic[1], actor, G, content_loss, style_loss, score))
                        self.writer.add_scalar(tag='alpha', scalar_value=float(alpha), global_step=total_step)
                        self.writer.add_scalar(tag='alpha_loss', scalar_value=float(alpha_loss), global_step=total_step)
                        self.writer.add_scalar(tag='actor', scalar_value=float(actor), global_step=total_step)
                        self.writer.add_scalar(tag='critic1', scalar_value=float(critic[0]), global_step=total_step)
                        self.writer.add_scalar(tag='critic2', scalar_value=float(critic[1]), global_step=total_step)

                state = s_
                total_step += 1
                steps += 1

                if global_episode <= cfg.EP_BOOTSTRAP and steps > 10:
                    print('global_ep: {}, memory len: {}'.format(global_episode, len(self.memories)))
                    global_episode += 1
                    break

                if (done or steps >= cfg.MAX_EP_STEPS) and global_episode >= cfg.EP_BOOTSTRAP:
                    if global_episode % 1000 == 0:
                    # if done or global_episode % 1000 == 0:
                        print('saving models!')
                        self.save_model(global_episode)
                    self.writer.add_scalar(tag='return', scalar_value=float(episode_reward), global_step=global_episode)
                    # self.writer.add_info(steps, episode_values, np.mean(scores))
                    global_episode += 1
                    rewards.append(np.mean(scores))
                    break

        self.writer.close()
        np.save('curve/reward_{}.npy'.format(seed), np.array(rewards))
        toc = time.time()
        time_elapse = toc - tic
        h = time_elapse // 3600
        m = time_elapse % 3600 // 60
        s = time_elapse % 3600 % 60
        print('cast time %.0fh %.0fm %.0fs' % (h, m, s))
        print('training finished!')

        return rewards

    def save_model(self, global_ep):
        print("------------------------Saving model------------------")
        self.brain.save_model(global_ep, cfg.MODEL_PATH)
        print("------------------------Model saved!------------------")
