import copy
import os
import random
#
import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
#
from datetime import datetime
from env.environment import PortfolioEnv
from plot import add_curve, add_hline, save_plot
from pyfolio import timeseries
from torch.distributions.normal import Normal
from torch.distributions import Distribution
from torch.multiprocessing import Pipe, Lock

# init, train, validate, test
# https://github.services.devops.takamol.support/TianhongDai/reinforcement-learning-algorithms/blob/master/rl_algorithms/sac/sac_agent.py
class SAC:
    def __init__(self, load=False, q_lr=3e-4, p_lr=3e-4, log_std_min=-20, log_std_max=2,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=100,
                 layer1_size=256, layer2_size=256, device='cpu', state_type='only prices', eval_episodes=1,
                 djia_year=2019, reward_scale=1, repeat=0, figure_dir='plots/sac', checkpoint_dir='checkpoints/sac'):
        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        elif djia_year == 2012:
            self.intervals = self.env.get_intervals(train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05)

        self.max_size = max_size
        self.eval_episodes = eval_episodes
        self.gamma = gamma
        self.tau = tau
        self.q_lr = q_lr
        self.p_lr = p_lr
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.repeat = repeat  # iteration
        self.batch_size = batch_size
        self.device = device
        self.reward_scale = reward_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # build up the network that will be used.
        self.qf1 = flatten_mlp(self.env.state_shape()[0], self.layer1_size, self.env.action_shape()[0])
        self.qf2 = flatten_mlp(self.env.state_shape()[0], self.layer2_size, self.env.action_shape()[0])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network
        self.actor_net = tanh_gaussian_actor(self.env.state_shape()[0], self.env.action_shape()[0],
                                             self.layer1_size, self.log_std_min, self.log_std_max)
        # define the optimizer for them
        self.qf1_optim = T.optim.Adam(self.qf1.parameters(), lr=self.q_lr)
        self.qf2_optim = T.optim.Adam(self.qf2.parameters(), lr=self.q_lr)
        # the optimizer for the policy network
        self.actor_optim = T.optim.Adam(self.actor_net.parameters(), lr=self.p_lr)
        # entropy target
        self.target_entropy = -np.prod(self.env.state_shape()).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        # define the optimizer
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=self.p_lr)
        # define the replay buffer
        self.buffer = replay_buffer(self.max_size)
        # get the action max
        self.action_max = 2#self.env.action_space.high[0]  # 2 or 2000?
        # if use cuda, put tensor onto the gpu
        if self.device == 'cuda':
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()

        self.figure_dir = figure_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if load:
            self.actor_net.load_state_dict(T.load(self.checkpoint_dir + f'/actor_net_{round}_{epoch}.pt'))

        np.random.seed(0)

    # train the agent
    def learn(self, round):
        global_timesteps = 0
        max_wealth = 0
        validation_history = []
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy='gaussian')
        # reset the environment
        obs = self.env.reset(*self.intervals['training'])
        # epochs
        for epoch in range(300):
            # length of each epoch
            # for t in range(self.batch_size):
            #     print(t)
                # start to collect samples
            with T.no_grad():
                obs_tensor = self._get_tensor_inputs(obs)
                pi = self.actor_net(obs_tensor)
                action = get_action_info(pi, cuda=self.device=='cuda').select_actions(reparameterize=False)
                action = action.cpu().numpy()[0]
            # input the actions into the environment
            obs_, reward, done, info, wealth = self.env.step(action)#self.action_max *
            # store the samples
            self.buffer.add(obs, action, reward, obs_, float(done))
            # reassign the observations
            obs = obs_
            if done:
                # reset the environment
                obs = self.env.reset(*self.intervals['training'])
                break
            # after collect the samples, start to update the network
            for _ in range(100):
                qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_network()
                # update the target network
                if global_timesteps % 1 == 0:
                    self._update_target_network(self.target_qf1, self.qf1)
                    self._update_target_network(self.target_qf2, self.qf2)
                global_timesteps += 1
            # print the log information
            if epoch % 5 == 0:
                # start to do the evaluation
                validation_wealth = self._evaluate_agent()
                validation_history.append(validation_wealth - 1000000)
                print('\n{} Epoch: {}, Wealth: {:,.0f} QF1: {:,.2f}, QF2: {:,.2f}, AL: {:,.2f}, Alpha: {:,.5f}, AlphaL: {:,.2f}'.format(
                        datetime.now(), epoch, validation_wealth, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))
                # save model if validation is creating new max wealth
                if validation_wealth > max_wealth:
                    T.save(self.actor_net.state_dict(), self.checkpoint_dir + f'/actor_net_{round}_{epoch}.pt')
                    # self.agent.save_models(round, saved_iter, self.checkpoint_dir)
                max_wealth = max(max_wealth, validation_wealth)
                # stop training if on validation period the last 5 iterations did not create a new maximum
                if validation_history[-5:].count(max_wealth - 1000000) != 1:
                    break


    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        obs = self.env.reset(*self.intervals['training'])
        for _ in range(100):
            if exploration_policy == 'uniform':
                raise NotImplementedError
            elif exploration_policy == 'gaussian':
                # the sac does not need normalize?
                with T.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    # generate the policy
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
                # input the action input the environment
                obs_, reward, done, info, wealth = self.env.step(self.action_max * action)
                # store the episodes
                self.buffer.add(obs, action, reward, obs_, float(done))
                obs = obs_
                if done:
                    # if done, reset the environment
                    obs = self.env.reset(*self.intervals['training'])
        print("Initial exploration has finished!")

    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0)
        return obs_tensor

    # update the network
    def _update_network(self):
        # smaple batch of samples from the replay buffer
        obses, actions, rewards, obses_, dones = self.buffer.sample(self.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        obses = T.tensor(obses, dtype=T.float32, device=self.device)
        actions = T.tensor(actions, dtype=T.float32, device=self.device)
        rewards = T.tensor(rewards, dtype=T.float32, device=self.device).unsqueeze(-1)
        obses_ = T.tensor(obses_, dtype=T.float32, device=self.device)
        inverse_dones = T.tensor(1 - dones, dtype=T.float32, device=self.device).unsqueeze(-1)
        # start to update the actor network
        pis = self.actor_net(obses)
        actions_info = get_action_info(pis, cuda=self.device=='cuda')
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        # use the automatically tuning
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # get the param
        alpha = self.log_alpha.exp()
        # get the q_value for new actions
        q_actions_ = T.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = ((alpha.clone() * log_prob.clone()) - q_actions_.clone()).mean()
        # policy loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        #clone, detach is used: https://github.com/NVlabs/FUNIT/issues/23
        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with T.no_grad():
            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.device=='cuda')
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = T.min(self.target_qf1(obses_, actions_next_),
                                            self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
            target_q_value = self.reward_scale * rewards + inverse_dones * self.gamma * target_q_value_next
        qf1_loss = (q1_value - target_q_value).pow(2).mean()
        qf2_loss = (q2_value - target_q_value).pow(2).mean()
        # qf1
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        # qf2
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()

    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # evaluate the agent
    def _evaluate_agent(self):
        total_reward = 0
        for _ in range(self.eval_episodes):
            obs = self.env.reset(*self.intervals['validation'])
            episode_reward = 0
            while True:
                with T.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi, cuda=self.device=='cuda').select_actions(exploration=False, reparameterize=False)
                    action = action.detach().cpu().numpy()[0]
                # input the action into the environment
                obs_, reward, done, info, wealth = self.env.step(action)#action only? self.action_max *
                episode_reward += reward
                if done:
                    break
                obs = obs_
            total_reward += episode_reward
        return wealth #total_reward / self.eval_episodes

    # backtest the agent
    def test(self):
        total_reward = 0
        for _ in range(self.eval_episodes):
            obs = self.env.reset(*self.intervals['testing'])
            episode_reward = 0
            while True:
                with T.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi, cuda=self.device=='cuda').select_actions(exploration=False,
                                                                               reparameterize=False)
                    action = action.detach().cpu().numpy()[0]
                # input the action into the environment
                obs_, reward, done, info, wealth = self.env.step(self.action_max * action)#action only?
                episode_reward += reward
                if done:
                    break
                obs = obs_
            total_reward += episode_reward
        return total_reward / self.eval_episodes

# the flatten mlp
class flatten_mlp(nn.Module):
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


class replay_buffer:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

# define the policy network - tanh gaussian policy network
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
        return (mean, T.exp(log_std))


# get action_infos
class get_action_info:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = T.tanh(self.mean)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)

"""
the tanhnormal distributions from rlkit may not stable
"""
class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return T.tanh(z), z
        else:
            return T.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = T.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - T.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return T.tanh(z), z
        else:
            return T.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = T.zeros(self.normal_mean.size(), dtype=T.float32)
        sample_std = T.ones(self.normal_std.size(), dtype=T.float32)
        z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
        z.requires_grad_()
        if return_pretanh_value:
            return T.tanh(z), z
        else:
            return T.tanh(z)