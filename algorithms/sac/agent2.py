import os
from env.environment import PortfolioEnv
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import multiprocessing as mp
from plot import add_curve, save_plot



# Agent, Replay Buffer, CriticNetwork, ActorNetwork, ValueNetwork
class Agent:
    def __init__(self, input_dims, n_actions, log_std_min, log_std_max, q_lr=3e-4, p_lr=3e-4,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=100,
                 layer1_size=256, layer2_size=256, device='cuda', reward_scale=1):




