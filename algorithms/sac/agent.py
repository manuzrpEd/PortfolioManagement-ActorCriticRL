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
    def __init__(self, input_dims, n_actions, alpha=0.001, beta=0.001,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=100,
                 layer1_size=256, layer2_size=256, cuda_index=0, reward_scale=1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions,
                                  name='actornetwork', max_action=2000,
                                  fc1_dims=layer1_size, fc2_dims=layer2_size, cuda_index=cuda_index)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_1', fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      cuda_index=cuda_index)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2', fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      cuda_index=cuda_index)
        self.value = ValueNetwork(beta, input_dims, name='value', cuda_index=cuda_index)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', cuda_index=cuda_index)

        self.scale = reward_scale
        self.update_network_parameters()

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self, round, iteration, address):
        print('... saving models ...')
        self.actor.save_checkpoint(round, iteration, address)
        self.value.save_checkpoint(round, iteration, address)
        self.target_value.save_checkpoint(round, iteration, address)
        self.critic_1.save_checkpoint(round, iteration, address)
        self.critic_2.save_checkpoint(round, iteration, address)

    def load_models(self, round, iteration, address):
        print('... loading models ...')
        self.actor.load_checkpoint(round, iteration, address)
        self.value.load_checkpoint(round, iteration, address)
        self.target_value.load_checkpoint(round, iteration, address)
        self.critic_1.load_checkpoint(round, iteration, address)
        self.critic_2.load_checkpoint(round, iteration, address)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale * reward + self.gamma * value_

        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, *n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self,  beta, input_dims, n_actions, name,
                 fc1_dims=256, fc2_dims=256, cuda_index=0, chkpt_dir='checkpoints/sac'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.name = name

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.checkpoint_dir = chkpt_dir
        self.name = name

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:{}'.format(str(cuda_index)) if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}_{round}_{iteration}')

    def load_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}_{round}_{iteration}'))


# %%
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, name, fc1_dims=256, fc2_dims=256,
                 cuda_index=0, chkpt_dir='checkpoints/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.checkpoint_dir = chkpt_dir
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:{}'.format(str(cuda_index)) if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v

    def save_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}_{round}_{iteration}')

    def load_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}_{round}_{iteration}'))


# %%
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, max_action=2000,
                 fc1_dims=256, fc2_dims=256, cuda_index=0, chkpt_dir='checkpoints/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.name = name
        self.max_action = 2
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, *self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, *self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:{}'.format(str(cuda_index)) if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state.float())
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        # mu, sigma = self.forward(state)
        # probabilities = Normal(mu, sigma)
        #
        # if reparameterize:
        #     actions = probabilities.rsample()  # sample with noise
        # else:
        #     actions = probabilities.sample()
        #
        # action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        # log_probs = probabilities.log_prob(actions)  # for calculation of loss function. not used for calculating which action to take
        # log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)  # From paper's appendix
        # log_probs = log_probs.sum(1, keepdim=True)
        #
        # return action, log_probs
        mean, logstd = self(state)

        cov = T.diag_embed(T.exp(logstd))
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()

        # if mean.shape[0] == 1:
        #     print('    policy entropy: ', dist.entropy().detach().cpu())
        #     print('    policy mean:    ', mean.detach().cpu())
        #     print('    policy std:     ', torch.exp(logstd).detach().cpu())
        #     print()

        ### Enforcing Action Bound
        action = T.tanh(u)
        logprob = dist.log_prob(u).unsqueeze(1) - T.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, logprob

    def save_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        T.save(self.state_dict(), f'{address}/{self.name}_{round}_{iteration}')

    def load_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(T.load(f'{address}/{self.name}_{round}_{iteration}'))