# https://github.com/JimmyYourHonor/GCL_SAC/blob/main/sac_torch.py
import os
import torch
#
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#
from torch.distributions import Distribution, Normal, MultivariateNormal

class Agent(object):
    def __init__(self, env, n_actions, input_dims,
                 alpha, beta, gamma, tau,
                 max_size, batch_size, fc1_dims, fc2_dims,
                 reward_scale):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, max_size, fc1_dims, fc2_dims, n_actions)#find out highest action?
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims , name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims , name='critic_2')
        self.value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=tau)

    def choose_action(self, observation):
        self.actor.eval()
        #reached
        # print("agent3 choose_action")
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=True)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in target_value_dict:
            target_value_dict[name] = tau * value_dict[name].clone() + \
                                      (1 - tau) * target_value_dict[name].clone()

        self.target_value.load_state_dict(target_value_dict)

    def save_models(self, round, iteration, address):
        print('... saving models ...')
        self.actor.save_checkpoint(round, iteration, address)
        self.critic_1.save_checkpoint(round, iteration, address)
        self.critic_2.save_checkpoint(round, iteration, address)
        self.value.save_checkpoint(round, iteration, address)
        self.target_value.save_checkpoint(round, iteration, address)

    def load_models(self, round, iteration, address):
        print('... loading models ...')
        self.actor.load_checkpoint(round, iteration, address)
        self.critic_1.load_checkpoint(round, iteration, address)
        self.critic_2.load_checkpoint(round, iteration, address)
        self.value.load_checkpoint(round, iteration, address)
        self.target_value.load_checkpoint(round, iteration, address)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, new_states, actions, rewards, dones = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)

        states_value = self.value(states).view(-1)
        new_states_value = self.target_value(new_states).view(-1)
        new_states_value[dones] = 0.0

        #reached
        # print("learn")
        action, log_probs = self.actor.sample_normal(states, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(states, action)
        q2_new_policy = self.critic_2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(states_value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        #reached repeated?
        # action, log_probs = self.actor.sample_normal(states, reparameterize=True)
        # log_probs = log_probs.view(-1)
        # q1_new_policy = self.critic_1(states, action)
        # q2_new_policy = self.critic_2(states, action)
        # critic_value = torch.min(q1_new_policy, q2_new_policy)
        # critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale * rewards + self.gamma * new_states_value
        q1_old_policy = self.critic_1(states, actions).view(-1)
        q2_old_policy = self.critic_2(states, actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        return value_loss, actor_loss, critic_loss

    # def train_on_env(self, env):
    #     rewards = []
    #     done = False
    #     observation = env.reset()
    #     while not done:
    #         action = self.choose_action(observation)
    #         observation_, reward, done, _ = env.step(action)
    #         self.remember(observation, action, reward, observation_, done)
    #         # if not load_checkpoints:
    #         self.learn()
    #         observation = observation_
    #         rewards.append(reward)
    #     return np.sum(rewards)
    #
    # def generate_session(self, env, t_max=1000):
    #     states, traj_probs, actions, rewards = [], [], [], []
    #     s = env.reset()
    #     q_t = 0
    #     for t in range(t_max):
    #         state = torch.Tensor([s]).to(self.actor.device)
    #         action, log_probs = self.actor.sample_normal(state, reparameterize=False)
    #         action = action.cpu().detach().numpy()[0]
    #
    #         new_s, r, done, info = env.step(action)
    #
    #         log_probs = log_probs.cpu().detach().numpy()[0]
    #         # q_t *= probs
    #         q_t += log_probs[0]
    #         states.append(s.tolist())
    #         traj_probs.append(q_t)
    #         actions.append(action[0])
    #         rewards.append(r)
    #
    #         s = new_s
    #         if done:
    #             break
    #
    #     return np.array(states), np.array(traj_probs), np.array(actions), np.array(rewards)

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims, fc2_dims, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_val = self.fc1(torch.cat([state, action], dim=1))
        action_val = self.bn1(action_val)
        action_val = F.relu(action_val)
        action_val = self.fc2(action_val)
        action_val = self.bn2(action_val)
        action_val = F.relu(action_val)

        q = self.q(action_val)
        return q

    def save_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        torch.save(self.state_dict(), f'{address}/{self.name}_{round}_{iteration}')

    def load_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(torch.load(f'{address}/{self.name}_{round}_{iteration}'))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.fc1(state.float())
        value = self.bn1(value)
        value = F.relu(value)
        value = self.fc2(value)
        value = self.bn2(value)
        value = F.relu(value)
        value = self.v(value)
        return value

    def save_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        torch.save(self.state_dict(), f'{address}/{self.name}_{round}_{iteration}')

    def load_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(torch.load(f'{address}/{self.name}_{round}_{iteration}'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims, fc2_dims, n_actions, name='actor',
                 chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)#1

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _forward(self, state):
        prob = self.fc1(state)
        prob = self.bn1(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = self.bn2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        if sigma.isnan().any():
            print("Errors handling forward state.")
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self._forward(state)
        try:
            probabilities = Normal(mu, sigma)
        except:
            print("Errors handling Normal probabilities of mu, sigma and state.")
        if (reparameterize):
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)

        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

    def save_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        torch.save(self.state_dict(), f'{address}/{self.name}_{round}_{iteration}')

    def load_checkpoint(self, round, iteration, address):
        if address is None:
            address = self.checkpoint_dir
        self.load_state_dict(torch.load(f'{address}/{self.name}_{round}_{iteration}'))

# This is the memory of the agent
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, size=batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, new_states, actions, rewards, dones

    def clear_buffer(self):
        self.mem_cntr = 0
