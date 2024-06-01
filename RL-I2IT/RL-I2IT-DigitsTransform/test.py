import torch
import random
import numpy as np
import copy
import cv2
import time

import utils
from dataloader import MnistData
from config import Config as cfg
from brain import SAC
from env import Env
from summary import Summary
from networks import *


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

if __name__ == "__main__":
    utils.setup_seed(cfg.SEED)
    utils.remkdir(cfg.TEST_PATH)

    #######################################
    stn = SpatialTransformer(cfg.HEIGHT, mode='bilinear').to(device)

    mnist = MnistData(cfg.MNIST_DIR, mode='test')

    brain = SAC(stn, device)
    brain.load_decoder(cfg.DECODER_MODEL_RL)
    brain.load_actor(cfg.ACTOR_MODEL)
    brain.load_critic(cfg.CRITIC1_MODEL, cfg.CRITIC2_MODEL)

    times = []
    i = 0

    while i < 50:
        # moving, fixed = mnist.next_same_class()
        moving, fixed_ = mnist.next_diff_class()
        # moving, fixed = mnist.read_test(i)

        if i % 1 == 0:
            cv2.imwrite('{}/{}-fixed.bmp'.format(cfg.TEST_PATH, i), fixed_)
            cv2.imwrite('{}/{}-moving.bmp'.format(cfg.TEST_PATH, i), moving)

        tic = time.time()

        moving = utils.tensor_im(moving).unsqueeze(0).to(device)
        fixed = utils.tensor_im(fixed_).unsqueeze(0).to(device)
        moved = copy.deepcopy(moving)

        pred = None
        step = 0
        while step < 30:
            state = torch.cat([fixed, moved], dim=1)
            latent, flow = brain.choose_action(state, test=True)
            pred = flow if pred is None else stn(pred, flow) + flow
            moved = stn(moving, pred)

            if step % 1 == 0:
                warped = utils.numpy_im(moved, device=device)
                cv2.imwrite('result/{}-{}-warped.bmp'.format(i, step), warped)
            step += 1

        toc = time.time()
        times.append(toc-tic)

        flow = utils.numpy(pred.squeeze(), device=device)

        warped = utils.numpy(stn(moving, pred).squeeze(), device=device)
        dice = utils.dice(fixed_>0, warped>0)[0]
        print('test: {}, dice: {:4f}'.format(i, dice))

        if i % 1 == 0:
            flow = utils.render_flow(flow)
            cv2.imwrite('result/{}-flow.png'.format(i), flow)

        i += 1

    print('avg time: {}'.format(np.mean(times)))
















