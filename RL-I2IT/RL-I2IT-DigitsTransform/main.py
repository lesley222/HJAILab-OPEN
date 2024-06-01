import torch
import random
import numpy as np
import copy
import torchvision

import utils
from config import Config as cfg
from brain import SAC
from env import Env
from agent import Agent
from summary import Summary
from networks import *
from dataloader import MnistData

# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')


if __name__ == "__main__":
    # utils.setup_seed(cfg.SEED)
    utils.mkdir(cfg.LOG_DIR)
    utils.remkdir(cfg.PROCESS_PATH)
    utils.mkdir(cfg.MODEL_PATH)
    utils.mkdir('curve')

    summary = Summary(cfg.LOG_DIR)

    #######################################
    stn = SpatialTransformer(cfg.HEIGHT, 'bilinear').to(device) # nearest
    mnist = MnistData(cfg.MNIST_DIR, mode='train')

    brain = SAC(stn, device)

    env = Env(mnist, stn, device)
    agent = Agent(brain, env, summary, device=device)

    rs = []
    for seed in np.random.randint(100, 1000, 1):

        rewards = agent.run(seed)
        rs.append(rewards)

    np.save('curve/mnist_sac.npy', np.array(rs))






