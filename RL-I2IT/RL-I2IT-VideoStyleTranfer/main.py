import argparse
import datetime
import os

import numpy as np
import itertools
import torch
from PIL import Image
from torch.utils.data import DataLoader
from datasets import SingleDataset
from torchvision import transforms
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from env import Env


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# environment
parser.add_argument('--env-name', default="impronte_d_artista",
                    help='environment (default: HalfCheetah-v2)')
parser.add_argument('--video_folder', default="../datasets/videos",
                    help='the dir of frames of videos')
parser.add_argument('--style_path', default="../datasets/style_images/impronte_d_artista.jpg",
                    help="the path of the style image")
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
# value setting
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=2e-4, metavar='G',
                    help='learning rate (default: 2e-4)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--env_batch_size', type=int, default=1, metavar='N',
                    help='batch size in env(default: 1)')
parser.add_argument('--sample_batch_size', type=int, default=1, metavar='N',
                    help='sample batch size in memory(default: 256)')
parser.add_argument('--max_episode_steps', type=int, default=10, metavar='N',
                    help='maximum number of steps each episode (default: 10)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--score_threshold', type=float, default=0, metavar='N',
                    help='maximum number of steps (default: 0)')
# state action
parser.add_argument('--state_size', type=int, default=256, metavar='N',
                    help='the size of state when input')
parser.add_argument('--state_input_dim', type=int, default=3, metavar='N',
                    help='the input dim of state')
parser.add_argument('--hidden_dim', type=int, default=16, metavar='N',
                    help='the number of kernel in hidden layer (default: 16)')
parser.add_argument('--action_dim', type=int, default=64, metavar='N',
                    help='the dim of action that give by policy')
# style transfer
parser.add_argument('--content_weight', type=int, default=1e0, metavar='N',
                    help='the weight of content loss')
parser.add_argument('--style_weight', type=int, default=1e5, metavar='N',
                    help='the weight of style loss')
parser.add_argument('--tv_weight', type=int, default=1e-7, metavar='N',
                    help='the weight of tv loss')
parser.add_argument('--temp_weight', type=int, default=1e2, metavar='N',
                    help='the weight of temporal loss')
# update iter
parser.add_argument('--start_episode', type=int, default=5, metavar='N',
                    help='Steps sampling random actions (default: 100)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--actor_update', type=bool, default=True, metavar='N',
                    help='Update or not update actor per no. of updates per step (default: True)')
parser.add_argument('--actor_update_interval', type=int, default=8, metavar='N',
                    help='actor update per no. of updates per step (default: 16)')
parser.add_argument('--critic_update', type=bool, default=True, metavar='N',
                    help='Update or not update critic per no. of updates per step (default: True)')
parser.add_argument('--critic_update_interval', type=int, default=4, metavar='N',
                    help='actor update per no. of updates per step (default: 16)')
parser.add_argument('--target_update_interval', type=int, default=8, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--vae_update', type=bool, default=True, metavar='N',
                    help='Update or not update vae per no. of updates per step (default: True)')
parser.add_argument('--vae_update_interval', type=int, default=4, metavar='N',
                    help='VAE update per no. of updates per step (default: 1)')

parser.add_argument('--replay_size', type=int, default=16000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--use_cuda', type=bool, default=True, metavar='N',
                    help='run on CUDA (default: False)')
parser.add_argument('--cuda_id', type=int, default=1, metavar='N',
                    help='run on no. of CUDA (default: 0)')
parser.add_argument('--continue_train', type=bool, default=False, metavar='N',
                    help='continue to train')
parser.add_argument('--checkpoint_dir', default="./checkpoints/",
                    help="the dir of checkpoint")

args = parser.parse_args()

# Environment
if args.use_cuda:
    device = torch.device("cuda", args.cuda_id)
else:
    device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize(args.state_size),
    transforms.CenterCrop(args.state_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = SingleDataset(args.video_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.env_batch_size, shuffle=False)
style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
style = style_transform(Image.open(args.style_path).convert('RGB')).unsqueeze(0)
env = Env(dataloader=dataloader, style=style, args=args, device=device)
# env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(style, args, device=device)
if args.continue_train:
    checkpoint_path = os.path.join(args.checkpoint_dir, 'sac_checkpoint_'+args.env_name+'_.ckpt')
    print('Loading weight from : ', checkpoint_path)
    agent.load_checkpoint(checkpoint_path)

# Tesnorboard
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                  args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# ----------------------------------------------------------------------------------------------------------------------
# Training Loop
total_numsteps = 0
updates = 0
current_video_name = None
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    finish_episode = False
    state, init_score, video_name = env.reset()  # tensor:[1, 3, x, x](device), int
    if current_video_name != video_name:
        agent.echo.init_hidden_state()
        current_video_name = video_name
    agent.policy.init_hidden_state()

    while not finish_episode:
        # env.render()
        if i_episode % 10 == 0:
            env.save_init('./process/')
        action, rec = agent.select_action(state, episode_steps)  # tensor:[1, 64, 64, 64], [](device)

        if len(memory) > args.sample_batch_size and args.start_episode < i_episode:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                agent.update_parameters(memory, args.sample_batch_size, updates, writer)
                updates += 1

        next_state, reward, done, score = env.step(rec)  # tensor(device), float, bool, float
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.max_episode_steps else float(not done)  # done->0
        finish_episode = True if episode_steps == env.max_episode_steps or done else False
        memory.push(state.detach().cpu().numpy()[0],
                    action.detach().cpu().numpy()[0],
                    reward,
                    next_state.detach().cpu().numpy()[0],
                    mask)  # Append transition to memory

        if episode_steps <= env.max_episode_steps and i_episode % 10 == 0:
            env.save_process(episode_steps, './process/')
        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))

    if i_episode % 100 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state, init_score, video_name = env.reset()
            episode_reward = 0
            done = False
            eval_step = 0
            while not done and eval_step < env.max_episode_steps:
                action, rec = agent.select_action(state, eval_step, evaluate=True)

                next_state, reward, done, _ = env.step(rec)
                episode_reward += reward
                eval_step += 1

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
    if i_episode % 100 == 0:
        print('save model!')
        agent.save_checkpoint(args.env_name, suffix=".ckpt")
