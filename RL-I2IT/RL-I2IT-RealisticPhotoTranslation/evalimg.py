import random
import numpy as np
import copy
import cv2
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

image_dir = './result'

pp, ss = [], []
for i in range(106):
    moving = cv2.imread('{}/{}_moving.png'.format(image_dir,i))
    target = cv2.imread('{}/{}_target.png'.format(image_dir, i))
    # print(moving)

    p = psnr(moving, target, data_range=255)
    s = ssim(moving, target, data_range=255, multichannel=True)

    pp.append(p)
    ss.append(s)
    print(p, s)

print('psnr mean: {}, std: {}'.format(np.mean(pp), np.std(pp)))
print('ssim mean: {}, std: {}'.format(np.mean(ss), np.std(ss)))