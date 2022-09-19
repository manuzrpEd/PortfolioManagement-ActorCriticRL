from env.environment import PortfolioEnv
from algorithms.sac2.agent import Agent, ActorNetwork, ValueNetwork, CriticNetwork
from torch.multiprocessing import Pipe, Lock
from plot import add_curve, add_hline, save_plot
import os
import pandas as pd
from pyfolio import timeseries

# init, train, validate, test
# https://github.services.devops.takamol.support/TianhongDai/reinforcement-learning-algorithms/blob/master/rl_algorithms/sac/sac_agent.py
class SAC:
    def __init__(self, load=False, alpha=0.001, beta=0.001,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=100,
                 layer1_size=256, layer2_size=256, cuda='cuda',
                 state_type='only prices', djia_year=2019, repeat=0):

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
        self.agent = Agent(input_dims=self.env.state_shape(), n_actions=self.env.action_shape(), alpha=alpha, beta=beta,
                 gamma=gamma, max_size=max_size, tau=tau, batch_size=batch_size,
                 layer1_size=layer1_size, layer2_size=layer2_size, cuda_index=cuda_index)
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