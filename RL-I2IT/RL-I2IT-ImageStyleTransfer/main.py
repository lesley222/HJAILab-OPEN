import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import utils
from config import Config as cfg
from brain import SAC
from env import Env
from agent import Agent
from datasets import Dataset, SingleDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from summary import Summary

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

if __name__ == "__main__":
    utils.mkdir(cfg.MODEL_PATH)
    utils.mkdir(cfg.CURVE_DIR)

    content_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),  # scale shortest side to image_size
        transforms.RandomCrop(cfg.IMAGE_SIZE),  # crop center image_size out
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])

    style_transform = transforms.Compose([
        # transforms.Resize(cfg.IMAGE_SIZE),  # scale shortest side to image_size
        # transforms.RandomCrop(cfg.IMAGE_SIZE),  # crop center image_size out
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])

    dataset = SingleDataset(cfg.DATASET_ROOT, transform=content_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    style = Image.open(cfg.STYLE_PATH).convert('RGB')
    style = style_transform(style).unsqueeze(0)

    env = Env(dataloader, style, device)

    summary = SummaryWriter('logs')
    #######################################
    rewards = []
    for seed in np.random.randint(100, 10000, 1):
        brain = SAC(device)

        agent = Agent(brain, env, summary, device=device)
        reward = agent.run(seed)
        rewards.append(reward)

    np.save(cfg.CURVE_DIR + '/sac_rewards.npy', np.array(rewards))
