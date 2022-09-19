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

# the flatten mlp
class flatten_mlp(nn.Module):
    #TODO: add the initialization method for it
    def __init__(self, input_dims, hidden_size, action_dims=None):
        super(flatten_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size) if action_dims is None else nn.Linear(input_dims + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, obs, action=None):
        inputs = T.cat([obs, action], dim=1) if action is not None else obs
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output

# define the policy network - tanh gaussian policy network
# TODO: Not use the log std
class tanh_gaussian_actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(tanh_gaussian_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = T.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        # the reparameterization trick
        # return mean and std
        return (mean, torch.exp(log_std))

# Agent, Replay Buffer, CriticNetwork, ActorNetwork, ValueNetwork
class Agent:
    def __init__(self, input_dims, n_actions, log_std_min, log_std_max, q_lr=3e-4, p_lr=3e-4,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=100,
                 layer1_size=256, layer2_size=256, device='cuda', reward_scale=1):
        # build up the network that will be used.
        self.qf1 = flatten_mlp(input_dims[0], layer1_size, n_actions[0])
        self.qf2 = flatten_mlp(input_dims[0], layer2_size, n_actions[0])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network
        self.actor_net = tanh_gaussian_actor(input_dims[0],n_actions[0],layer1_size,log_std_min,log_std_max)
        # define the optimizer for them
        self.qf1_optim = T.optim.Adam(self.qf1.parameters(), lr=q_lr)
        self.qf2_optim = T.optim.Adam(self.qf2.parameters(), lr=q_lr)
        # the optimizer for the policy network
        self.actor_optim = T.optim.Adam(self.actor_net.parameters(), lr=p_lr)
        # entropy target
        self.target_entropy = -np.prod(n_actions).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        # define the optimizer
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=p_lr)
        # define the replay buffer
        self.buffer = replay_buffer(max_size)
        # get the action max
        self.action_max = self.env.action_space.high[0]#2 or 2000?
        # if use cuda, put tensor onto the gpu
        if device=='cuda':
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()



