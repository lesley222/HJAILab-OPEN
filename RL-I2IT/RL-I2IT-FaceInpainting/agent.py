import torch
from torchvision import datasets, transforms
import numpy as np
import h5py
import time
import os
from collections import namedtuple

import utils
from config import Config as cfg
from memory import ReplayMemory

Memory = namedtuple('Memory', ['im', 'm', 's', 'a', 'r', 's_', 'd'])

class Agent:
    def __init__(self, brain, env, summary_writer, device=None):
        super(Agent, self).__init__()
        self.writer = summary_writer
        self.device = device
        self.brain = brain
        self.env = env

        self.memories = ReplayMemory(cfg.MEMORY_SIZE, look_forward_steps=0)

    def feed_memory(self, im, m, s, a, r, s_, d):
        memory = Memory(im, m, s, a, r, s_, d)
        self.memories.store(memory)

    def run(self):
        tic = time.time()
        print('start training!')

        total_step = 1
        global_ep = 0
        update_actor = False
        update_vae = False
        while global_ep < cfg.MAX_GLOBAL_EP:
            ep_reward = 0
            step = 0
            episode_values = []

            s, im, mask, init_score = self.env.reset()

            if global_ep % 100 == 0:
                self.env.save_init()
            while True:
                latent, field = self.brain.choose_action(s)
                v = self.brain.get_value(s, latent)
                r, s_, done, score = self.env.step(field)

                ep_reward = r if step == 0 else (ep_reward * 0.99 + r * 0.01)
                episode_values.append(v)

                if done:
                    print("EP: ", global_ep,
                          "\tReward = ", ep_reward,
                          "\tSteps = ", step,
                          '\tinit_score = ', init_score,
                          '\tdone_score = ', score,
                          '\tDone!')

                # print(score)
                if step % 2 == 0:

                    self.feed_memory(utils.numpy(im, device=self.device),
                                        utils.numpy(mask, device=self.device),
                                        utils.numpy(s, device=self.device),
                                        utils.numpy(latent, device=self.device), r,
                                        utils.numpy(s_, device=self.device), done)


                if total_step % (cfg.FREQUENCY_ACTOR*cfg.UPDATE_GLOBAL_ITER) == 0:
                    update_actor = True
                else:
                    update_actor = False

                if total_step % (cfg.FREQUENCY_VAE*cfg.UPDATE_GLOBAL_ITER) == 0:
                    update_vae = True
                else:
                    update_vae = False

                if global_ep >= cfg.EP_BOOTSTRAP and total_step % cfg.UPDATE_GLOBAL_ITER == 0:
                    if len(self.memories) < cfg.SAMPLE_BATCH_SIZE:
                        samples = self.memories.sample(len(self.memories))
                    else:
                        samples = self.memories.sample(cfg.SAMPLE_BATCH_SIZE)

                    loss = self.brain.optimize(global_ep, update_actor, update_vae, samples)

                    if update_actor:
                        critic = min(loss['critic1'].item(), loss['critic2'].item())
                        actor = loss['actor'].item()
                        alpha = loss['alpha'].item()
                        vae = loss['vae'].item()
                        if global_ep < cfg.FIRST_EP:
                            print('ep-{}-{}:'.format(global_ep, step),
                                ' ----- critic, actor, vae, reward, init_score: ',
                                '{:.2f}, {:.4f}, {:.3f}, {:.4f}, {:.4f}'.format(
                                critic, actor, vae, r, init_score))
                        else:
                            real = loss['real']
                            fake = loss['fake']

                            print('ep-{}-{}:'.format(global_ep, step),
                                ' ----- critic, vae, real, fake, reward, init_score: ',
                                '{:.2f}, {:.4f}, {:.3f}, {:.3f}, {:.4f}, {:.4f}'.format(
                                critic, vae, real, fake, r, init_score))


                if step < 31 and global_ep % 100 == 0:
                    self.env.save_process(step)
                s = s_
                total_step += 1
                step += 1

                if global_ep <= cfg.EP_BOOTSTRAP and step > 10:
                    print('global_ep: {}, memory len: {}'.format(global_ep, len(self.memories)))
                    global_ep += 1
                    break

                if (done or step >= cfg.MAX_EP_STEPS) and global_ep >= cfg.EP_BOOTSTRAP:
                    if global_ep % 10000 == 0:
                        self.save_model(global_ep)
                    self.writer.add_info(step, episode_values, ep_reward)
                    global_ep += 1
                    break

        self.writer.close()
        toc = time.time()
        time_elapse = toc - tic
        h = time_elapse // 3600
        m = time_elapse % 3600 // 60
        s = time_elapse % 3600 % 60
        print('cast time %.0fh %.0fm %.0fs' % (h, m, s))
        print('training finished!')

    def save_model(self, global_ep):
        if global_ep > 500 and global_ep % 100 == 0:
            print("------------------------Saving model------------------")
            self.brain.save_model(global_ep, cfg.MODEL_PATH)
            print("------------------------Model saved!------------------")



























