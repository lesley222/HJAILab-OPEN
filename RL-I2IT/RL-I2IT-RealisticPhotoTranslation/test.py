import torch
import random
import numpy as np
import copy
import cv2
import time
import torchvision.utils as vutils
# from skimage.measure import compare_ssim as ssim # Changed in version 0.16: This function was renamed from skimage.measure.compare_ssim to skimage.metrics.structural_similarity.
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_psnr as psnr # This function was renamed in release 0.16 and the older compare_psnr name was removed in 0.18. 
from skimage.metrics import peak_signal_noise_ratio as psnr

import utils
from dataloader import Dataset
from config import Config as cfg
from brain import SAC
from env import Env
from summary import Summary
from networks import *
from options.test_options import TestOptions
from data import create_dataset
import lpips
# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')


if __name__ == "__main__":
    utils.setup_seed(cfg.SEED)
    utils.remkdir(cfg.RESULT_PATH)

    #######################################
    # dataset = Dataset(cfg.HEIGHT, mode='test', path=cfg.TEST_PATH)

    # test_loader = torch.utils.data.DataLoader(dataset,
    #                                        batch_size=1,
    #                                        shuffle=False,
    #                                        num_workers=0, drop_last=False)

    opt = TestOptions().parse()   # get training options
    opt.no_flip = True
    opt.serial_batches = True
    opt.dataset_mode = 'aligned'
    dataset = create_dataset(opt)

    # test_loader = test_data.generator()
    brain = SAC(cfg.HEIGHT, device)
    # brain.load_decoder(cfg.DECODER_MODEL_RL)
    # brain.load_actor(cfg.ACTOR_MODEL)
    # brain.load_critic(cfg.CRITIC1_MODEL, cfg.CRITIC2_MODEL)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    pp, ss, lps = [], [], []
    ll1=0
    ll2=0
    size = 0
    times = []
    hist_perframe = np.zeros((19, 19))
    for i, item in enumerate(dataset):
        target = item['A']
        source = item['B']

        tic = time.time()

        target = target.to(device)
        source = source.to(device)

        # moving = copy.deepcopy(source)
        moving = torch.randn(*source.shape).to(device)
        state = torch.cat([source, moving], dim=1)

        pred = None
        step = 0
        while step < 5:
            # state = torch.cat([im_fixed, mask], dim=1)
            latent, moving = brain.choose_action(state, test=False)
            state = torch.cat([source, moving], dim=1)

            step += 1

        toc = time.time()
        times.append(toc-tic)

        p = utils.psnr(
            utils.numpy_im(target, device),
            utils.numpy_im(moving, device))

        s = ssim(utils.numpy_im(target, device),
                 utils.numpy_im(moving, device), 
                 data_range=255, multichannel=True, win_size=3)

        lp = loss_fn(target, moving).squeeze().item()

        print(p, s, lp)
        pp.append(p)
        ss.append(s)
        lps.append(lp)

        if i <= 109:
            vutils.save_image(moving.data, cfg.RESULT_PATH+'/{}_moving.png'.format(i), normalize=True)
            vutils.save_image(source.data, cfg.RESULT_PATH+'/{}_source.png'.format(i), normalize=True)
            vutils.save_image(target.data, cfg.RESULT_PATH+'/{}_target.png'.format(i), normalize=True)
            # for idx in range(len(moving)):
            #     vutils.save_image(target[idx],'result/{}_{}_val_target_samples.png'.format(i, idx),normalize=True)
            #     vutils.save_image(source[idx].data,'result/{}_{}_val_source_samples.png'.format(i, idx),normalize=True)
            #     vutils.save_image(moving[idx].data,'result/{}_{}_val_predict_samples.png'.format(i, idx),normalize=True)

    print('avg time: {},\npsnr mean: {}, std: {}'
            .format(np.mean(times), np.mean(pp), np.std(pp)))
    print('ssim mean: {}, std: {}'.format(np.mean(ss), np.std(ss)))

    print('lpips mean: {}, std: {}'.format(np.mean(lps), np.std(lps)))
















