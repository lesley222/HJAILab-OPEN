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

Memory = namedtuple('Memory', ['t', 's', 'a', 'r', 's_', 'd'])

class Agent:
    def __init__(self, brain, env, summary_writer, device=None):
        super(Agent, self).__init__()
        self.writer = summary_writer
        self.device = device
        self.brain = brain
        self.env = env

        self.memories = ReplayMemory(cfg.MEMORY_SIZE, look_forward_steps=0)

    def feed_memory(self, t, s, a, r, s_, d):
        memory = Memory(t, s, a, r, s_, d)
        self.memories.store(memory)

    def run(self, seed):
        utils.setup_seed(seed)
        tic = time.time()
        print('start training!')

        total_step = 1
        global_ep = 0
        update_actor = False
        update_vae = False
        rewards = []
        while global_ep < cfg.MAX_GLOBAL_EP:
            ep_reward = 0
            step = 0
            episode_values = []
            scores = []
            s, target, init_score = self.env.reset()

            if global_ep % 100 == 0:
                self.env.save_init()
            while True:
                latent, field = self.brain.choose_action(s)
                v = self.brain.get_value(s, latent)
                r, s_, done, score = self.env.step(field)
                scores.append(score)

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
                if step % 3 == 0:
                    self.feed_memory(utils.numpy(target, device=self.device),
                                        utils.numpy(s, device=self.device),
                                        utils.numpy(latent, device=self.device),
                                        r,
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
                        critic = loss['critic1'].item(), loss['critic2'].item()
                        actor = loss['actor'].item()
                        alpha = loss['alpha'].item()
                        vae = loss['vae'].item()
                        if global_ep < cfg.FIRST_EP:
                            print('ep-{}-{}:'.format(global_ep, step),
                                ' ----- critic1, critic2, actor, vae, reward, init_score: ',
                                '{:.2f}, {:.4f}, {:.4f}, {:.3f}, {:.4f}, {:.4f}'.format(
                                critic[0], critic[1], actor, vae, score, init_score))
                        else:
                            real = loss['real']
                            fake = loss['fake']

                            print('ep-{}-{}:'.format(global_ep, step),
                                ' ----- critic1, critic2, vae, real, fake, reward, init_score: ',
                                '{:.2f}, {:.4f}, {:.4f}, {:.3f}, {:.3f}, {:.4f}, {:.4f}'.format(
                                critic[0], critic[1], vae, real, fake, score, init_score))


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
                    # if global_ep % 5000 == 0:
                    #     self.save_model(global_ep)
                    self.writer.add_info(step, episode_values, np.mean(scores))
                    global_ep += 1
                    rewards.append(np.mean(scores))
                    break

        self.writer.close()
        np.save(cfg.NAME+'_curve/reward_{}.npy'.format(seed), np.array(rewards))
        toc = time.time()
        time_elapse = toc - tic
        h = time_elapse // 3600
        m = time_elapse % 3600 // 60
        s = time_elapse % 3600 % 60
        print('cast time %.0fh %.0fm %.0fs' % (h, m, s))
        print('training finished!')

        return rewards

    def save_model(self, global_ep):
        if global_ep > 500 and global_ep % 100 == 0:
            print("------------------------Saving model------------------")
            self.brain.save_model(global_ep, cfg.MODEL_PATH)
            print("------------------------Model saved!------------------")



























