import torch
import random
import numpy as np
import copy
import torchvision

import utils
from dataloader import Dataset
from config import Config as cfg
from brain import SAC
from env import Env
from agent import Agent
from summary import Summary
from networks import *
from options.train_options import TrainOptions
from data import create_dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    # devices = []
    # for i in range(cfg.N_WORKERS):
    #     devices.append(torch.device('cuda:{}'.format(cfg.GPUS[i])))
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')


if __name__ == "__main__":
    # utils.setup_seed(cfg.SEED)
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)

    env = Env(dataset, device)

    utils.mkdir(cfg.LOG_DIR)
    utils.remkdir(cfg.PROCESS_PATH)
    utils.mkdir(cfg.MODEL_PATH)
    summary = Summary(cfg.LOG_DIR)

    #######################################
    curve_dir = cfg.NAME + '_curve'
    utils.mkdir(curve_dir)

    rs = []
    for seed in np.random.randint(100, 10000, 3):
        brain = SAC(cfg.HEIGHT, device)

        agent = Agent(brain, env, summary, device=device)
        reward = agent.run(seed)
        rs.append(reward)

    np.save(curve_dir+'/sac_rewards.npy', np.array(rs))






