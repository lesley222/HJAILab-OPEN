import torch
import random
import numpy as np
import copy
import cv2
import time
import torchvision.utils as vutils
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr # This function was renamed in release 0.16 and the older compare_psnr name was removed in 0.18. 
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import utils
from dataloader import Celeba, CelebaHQ
from config import Config as cfg
from brain import SAC
from env import Env
from summary import Summary
from networks import *

# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')


if __name__ == "__main__":
    utils.setup_seed(cfg.SEED)
    utils.remkdir(cfg.TEST_PATH)

    #######################################
    if cfg.DATASET == 'celeba':
        dataset = Celeba(cfg.HEIGHT, hole_size=cfg.HOLE_SIZE, mode='test')
    elif cfg.DATASET == 'celeba-hq':
        # path = 'F:\\0 服务器备份\\0 数据集\\CelebA_HQ' # windows path
        dataset = CelebaHQ(cfg.HEIGHT, hole_size=cfg.HOLE_SIZE, mode='test')

    test_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=0, drop_last=False)

    # test_loader = test_data.generator()
    brain = SAC(cfg.HEIGHT, device)
    brain.load_decoder(cfg.DECODER_MODEL_RL)
    brain.load_actor(cfg.ACTOR_MODEL)
    # brain.load_critic(cfg.CRITIC1_MODEL, cfg.CRITIC2_MODEL)

    pp=0
    ll1=0
    ll2=0
    ss=0
    size = 0
    times = []
    for i, (im_cut, im, mask, x, y) in enumerate(test_loader):
        # if i > 40:
        #     break
        # if i <= 3:
        #     vutils.save_image(im_cut.data, 'result/{}_im_cut.png'.format(i), normalize=True)
        #     vutils.save_image(im.data, 'result/{}_im.png'.format(i), normalize=True)

        tic = time.time()

        im_cut = im_cut.to(device)
        im = im.to(device)
        mask = mask.to(device)
        center = utils.get_area(im, cfg.HOLE_SIZE)

        im_fixed = copy.deepcopy(im_cut)

        pred = None
        step = 0
        while step < 15:
            # state = torch.cat([im_fixed, mask], dim=1)
            latent, flow = brain.choose_action(im_fixed, test=True)
            # pred = flow if pred is None else pred + flow
            im_fixed = utils.merge_area(im_fixed, flow)
            step += 1

        toc = time.time()
        times.append(toc-tic)

        if i <= 10:
            # vutils.save_image(flow.data, 'result/{}_flow.png'.format(i), normalize=True)
            # vutils.save_image(im_fixed.data, 'result/{}_fake.png'.format(i), normalize=True)
            for idx in range(len(im)):
                vutils.save_image(im[idx],'result/{}_{}_val_real_samples.png'.format(i, idx),normalize=True)
                vutils.save_image(im_cut[idx].data,'result/{}_{}_val_cropped_samples.png'.format(i, idx),normalize=True)
                vutils.save_image(im_fixed[idx].data,'result/{}_{}_val_recon_samples.png'.format(i, idx),normalize=True)

        fake = im_fixed.data.cpu().numpy()
        real_center = im.data.cpu().numpy()

        real_center = (real_center+1)*127.5
        fake = (fake+1)*127.5

        t = real_center - fake


        l1=0
        l2=0

        s=0
        p=0
        for j in range(len(fake)):
            p = p + psnr(real_center[j].transpose(1,2,0) , fake[j].transpose(1,2,0), data_range=255)
            s = s + ssim(real_center[j].transpose(1,2,0), fake[j].transpose(1,2,0), multichannel=True, win_size=3, data_range=255.0)
            l1 += np.mean(np.abs(t[j]))
            l2 += np.mean(np.square(t[j]))

        ss += s
        pp += p
        ll1 += l1
        ll2 += l2
        size += len(fake)
        print(i, p/len(fake), s/len(fake), l1 / len(fake), l2/len(fake))

    print('psnr: {}, ssim: {}, l1: {}, l2: {},'.format(pp / size, ss / size, ll1/size, ll2/size))

    print('avg time: {}'.format(np.mean(times)))
















