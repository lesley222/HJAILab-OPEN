import os
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from config import Config as cfg
import utils

from brain import SAC
from vgg import Vgg16
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim


def style_transfer(content_path, style_path, steps):
    # GPU enabling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.TEST_GPU_ID != -1:
        torch.cuda.set_device(0)
        print("Current device: %d" % torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True, parents=True)

    # load style model
    brain = SAC(device)
    brain.load_actor(cfg.encoder_model_path)
    brain.load_decoder(cfg.decoder_model_path)

    # content image
    content_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),  # scale shortest side to image_size
        transforms.CenterCrop(cfg.IMAGE_SIZE),  # crop center image_size out
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])

    style_transform = transforms.Compose([
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])

    content_image = Image.open(content_path).convert('RGB')
    style_image = Image.open(style_path)

    image_A = content_transform(content_image).unsqueeze(0)
    image_B = style_transform(style_image).unsqueeze(0)
    utils.save_image(os.path.join(output_dir, 'content.jpg'), image_A.data[0])
    utils.save_image(os.path.join(output_dir, 'style.jpg'), image_B.data[0])

    print('test:')
    content = image_A.to(device)

    # process input image
    state = content
    for step in range(1, steps+1):
        print('step=', step)
        _, stylized = brain.choose_action(state)
        state = stylized
        if step % 1 == 0:
            utils.save_image(os.path.join(output_dir, '{}_moving.jpg'.format(step)), stylized.cpu().data[0])

    print('finish!')


if __name__ == '__main__':
    content_path = './figures/golden_gate.jpg'
    style_path = './figures/blue_swirls.jpg'
    style_transfer(content_path=content_path, style_path=style_path, steps=10)
