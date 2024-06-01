import torch
import random
import numpy as np
import copy
import torchvision

import utils
from dataloader import Celeba, CelebaHQ
from config import Config as cfg
from brain import SAC
from env import Env
from agent import Agent
from summary import Summary
from networks import *

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

    summary = Summary(cfg.LOG_DIR)

    #######################################
    if cfg.DATASET == 'celeba':
        dataset = Celeba(cfg.HEIGHT, hole_size=cfg.HOLE_SIZE, mode='train')
    elif cfg.DATASET == 'celeba-hq':
        # path = 'F:\\0 服务器备份\\0 数据集\\CelebA_HQ' # windows path
        dataset = CelebaHQ(cfg.HEIGHT, hole_size=cfg.HOLE_SIZE, mode='train')

    brain = SAC(cfg.HEIGHT, device)
    if cfg.PRE_TRAIN:
        brain.load_decoder(cfg.DECODER_MODEL_RL)
        brain.load_actor(cfg.ACTOR_MODEL)
        brain.load_critic(cfg.CRITIC1_MODEL, cfg.CRITIC2_MODEL)
        # brain.load_netD(cfg.NETD_MODEL)

    env = Env(dataset, device)
    agent = Agent(brain, env, summary, device=device)

    agent.run()






