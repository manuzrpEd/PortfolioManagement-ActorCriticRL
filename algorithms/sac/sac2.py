from env.environment import PortfolioEnv
from algorithms.sac2.agent import Agent, ActorNetwork, ValueNetwork, CriticNetwork
from torch.multiprocessing import Pipe, Lock
from plot import add_curve, add_hline, save_plot
import os
import pandas as pd
from pyfolio import timeseries

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

# init, train, validate, test
# https://github.services.devops.takamol.support/TianhongDai/reinforcement-learning-algorithms/blob/master/rl_algorithms/sac/sac_agent.py
class SAC:
    def __init__(self, load=False, q_lr=3e-4, p_lr=3e-4,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=100,
                 layer1_size=256, layer2_size=256, cuda='cuda',
                 state_type='only prices', djia_year=2019, reward_scale=1, repeat=0):

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # build up the network that will be used.
        self.qf1 = flatten_mlp(self.env.state_shape()[0], layer1_size, self.env.action_shape()[0])
        self.qf2 = flatten_mlp(self.env.state_shape()[0], layer2_size, self.env.action_shape()[0])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network
        self.actor_net = tanh_gaussian_actor(self.env.state_shape()[0], self.env.action_shape()[0], layer1_size, log_std_min, log_std_max)
        # define the optimizer for them
        self.qf1_optim = T.optim.Adam(self.qf1.parameters(), lr=q_lr)
        self.qf2_optim = T.optim.Adam(self.qf2.parameters(), lr=q_lr)
        # the optimizer for the policy network
        self.actor_optim = T.optim.Adam(self.actor_net.parameters(), lr=p_lr)
        # entropy target
        self.target_entropy = -np.prod(self.env.state_shape()).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        # define the optimizer
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=p_lr)
        # define the replay buffer
        self.buffer = replay_buffer(max_size)
        # get the action max
        self.action_max = self.env.action_space.high[0]  # 2 or 2000?
        # if use cuda, put tensor onto the gpu
        if device == 'cuda':
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()

        self.figure_dir = 'plots/sac'
        self.checkpoint_dir = 'checkpoints/sac'
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.repeat = repeat # iteration
        self.batch_size = batch_size
        self.device = cuda

        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        elif djia_year == 2012:
            self.intervals = self.env.get_intervals(train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05)

        if load:
            self.agent.load_models(self.checkpoint_dir)

        np.random.seed(0)

    # train the agent
    def learn(self):
        global_timesteps = 0
        max_wealth = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy='gaussian')
        # reset the environment
        obs = self.env.reset(*self.intervals['training'])
        # epochs
        for epoch in range(300):
            # loops per epoch
            for _ in range(1):
                # length of each epoch
                for t in range(self.batch_size):
                    # start to collect samples
                    with T.no_grad():
                        obs_tensor = self._get_tensor_inputs(obs)
                        pi = self.actor_net(obs_tensor)
                        action = get_action_info(pi, cuda=self.device).select_actions(reparameterize=False)
                        action = action.cpu().numpy()[0]
                    # input the actions into the environment
                    obs_, reward, done, _ = self.env.step(self.action_max * action)
                    # store the samples
                    self.buffer.add(obs, action, reward, obs_, float(done))
                    # reassign the observations
                    obs = obs_
                    if done:
                        # reset the environment
                        obs = self.env.reset()
                # after collect the samples, start to update the network
                for _ in range(100):
                    qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork()
                    # update the target network
                    if global_timesteps % 1 == 0:
                        self._update_target_network(self.target_qf1, self.qf1)
                        self._update_target_network(self.target_qf2, self.qf2)
                    global_timesteps += 1
            # print the log information
            if epoch % 1 == 0:
                # start to do the evaluation
                validation_wealth = self._evaluate_agent()
                print('[{}] Epoch: {}, Rewards: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.5f}, AlphaL: {:.5f}'.format(
                        datetime.now(), epoch, validation_wealth, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))
                # save model if validation is creating new max wealth
                if validation_wealth > max_wealth:
                    saved_iter = iteration
                    self.agent.save_models(round, saved_iter, self.checkpoint_dir)
                max_wealth = max(max_wealth, validation_wealth)
                # stop training if on validation period the last 5 iterations did not create a new maximum
                if validation_history[-5:].count(max_wealth - 1000000) != 1:
                    break
                T.save(self.actor_net.state_dict(), self.checkpoint_dir + f'/model_{self.repeat}_{epoch}.pt')

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
                obs_, reward, done, _ = self.env.step(self.action_max * action)
                # store the episodes
                self.buffer.add(obs, action, reward, obs_, float(done))
                obs = obs_
                if done:
                    # if done, reset the environment
                    obs = self.env.reset()
        print("Initial exploration has finished!")

    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = T.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return obs_tensor

    # update the network
    def _update_newtork(self):
        # smaple batch of samples from the replay buffer
        obses, actions, rewards, obses_, dones = self.buffer.sample(self.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        obses = torch.tensor(obses, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        obses_ = torch.tensor(obses_, dtype=torch.float32, device=device)
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32, device=device).unsqueeze(-1)
        # start to update the actor network
        pis = self.actor_net(obses)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
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
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()
        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with T.no_grad():
            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obses_, actions_next_),
                                            self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
            target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next
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
        # policy loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()

    def train(self, round, verbose=False):
        training_history = []
        validation_history = []
        iteration = 1
        max_wealth = 0

        while True:
            observation = self.env.reset(*self.intervals['training'])
            done = False
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info, wealth = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, int(done))
                self.agent.learn()
                observation = observation_
                if verbose:
                    print(f"SAC training - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                          f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            # self.agent.memory.clear_buffer()

            print(f"SAC training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000 :,}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(
                f"SAC validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000 :,}")
            validation_history.append(validation_wealth - 1000000)
            # save model if validation is creating new max wealth
            if validation_wealth > max_wealth:
                saved_iter = iteration
                self.agent.save_models(round, saved_iter, self.checkpoint_dir)
            max_wealth = max(max_wealth, validation_wealth)
            # stop training if on validation period the last 5 iterations did not create a new maximum
            if validation_history[-5:].count(max_wealth - 1000000) != 1:
                break
            # stop training if iteration is equal to some number
            if iteration == 100:
                break
            iteration += 1

        self.agent.load_models(round, saved_iter, self.checkpoint_dir)

        buy_hold_history = self.env.buy_hold_history(*self.intervals['training'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(training_history, 'SAC')
        save_plot(filename=self.figure_dir + f'/{self.repeat}0_training.png',
                  title=f"Training - {self.intervals['training'][0].date()} to {self.intervals['training'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

        buy_hold_history = self.env.buy_hold_history(*self.intervals['validation'])
        buy_hold_final = (buy_hold_history[-1] / buy_hold_history[0] - 1) * 1000000
        add_hline(buy_hold_final, 'Buy&Hold')
        add_curve(validation_history, 'SAC')
        save_plot(filename=self.figure_dir + f'/{self.repeat}1_validation.png',
                  title=f"Validation - {self.intervals['validation'][0].date()} to {self.intervals['validation'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
            if verbose:
                print(f"SAC validation - Date: {info.date()},\tBalance: {int(self.env.get_balance()) :,},\t"
                      f"Cumulative Return: {int(wealth) - 1000000 :,},\tShares: {self.env.get_shares()}")
        return wealth

    def test(self, verbose=False):
        return_history = [0]
        buy_hold_history = self.env.buy_hold_history(*self.intervals['testing'])
        add_curve((buy_hold_history / buy_hold_history[0] - 1) * 1000000, 'Buy&Hold')

        observation = self.env.reset(*self.intervals['testing'])
        wealth_history = [self.env.get_wealth()]
        done = False
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            self.agent.remember(observation, action, reward, observation_, int(done))
            self.agent.learn()
            observation = observation_
            if verbose:
                print(f"SAC testing - Date: {info.date()},\tBalance: {int(self.env.get_balance()) :,},\t"
                      f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
            return_history.append(wealth - 1000000)
            wealth_history.append(wealth)
        # self.agent.memory.clear_buffer()

        add_curve(return_history, 'SAC')
        save_plot(self.figure_dir + f'/{self.repeat}2_testing.png',
                  title=f"Testing - {self.intervals['testing'][0].date()} to {self.intervals['testing'][1].date()}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

        returns = pd.Series(wealth_history, buy_hold_history.index).pct_change().dropna()
        stats = timeseries.perf_stats(returns)
        stats.to_csv(self.figure_dir + f'/{self.repeat}3_perf.csv')