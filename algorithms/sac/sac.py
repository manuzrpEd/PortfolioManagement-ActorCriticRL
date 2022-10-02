import os
#
import pandas as pd
#
from algorithms.sac.agent import Agent, ActorNetwork, CriticNetwork, ValueNetwork, ReplayBuffer#, LinearVAE, ActorNetwork_2
from env.environment import PortfolioEnv
from plot import add_curve, add_hline, save_plot
from pyfolio import timeseries

# init, train, validate, test
class SAC:
    def __init__(self, load=False, alpha=0.0001, beta=0.0001, gamma=0.99, tau=0.005,
                 max_size=1000000, reward_scale=2,
                 batch_size=64, layer1_size=1024, layer2_size=1024, t_max=256,
                 state_type='only prices', djia_year=2019, repeat=0):

        self.figure_dir = 'plots/sac'
        self.checkpoint_dir = 'checkpoints/sac'
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.t_max = t_max
        self.repeat = repeat

        self.env = PortfolioEnv(action_scale=1000, state_type=state_type, djia_year=djia_year)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
        elif djia_year == 2012:
            self.intervals = self.env.get_intervals(train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05)
        self.agent = Agent(env=self.env, n_actions=self.env.action_shape()[0], input_dims=self.env.state_shape()[0],
                           alpha=alpha, beta=beta, gamma=gamma, tau=tau,
                           max_size=max_size, batch_size=batch_size,
                           fc1_dims=layer1_size, fc2_dims=layer2_size,
                           reward_scale=reward_scale)
        if load:
            self.agent.load_models(self.checkpoint_dir)

    def train(self, round, verbose=False):
        training_history = []
        validation_history = []
        iteration = 1
        max_wealth = 0

        while True:
            observation = self.env.reset(*self.intervals['training'])
            done = False
            while not done:
                #reached
                # print("sac3 action")
                action = self.agent.choose_action(observation)
            #     action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, info, wealth = self.env.step(action)
            #     observation_, reward, done, info, wealth = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, done)
            #     self.agent.remember(observation, action, prob, val, reward, done)
                self.agent.learn()
                observation = observation_
                if verbose:
                    print(f"SAC training - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                          f"Cumulative Return: {int(wealth) - 1000000 :,},\tShares: {self.env.get_shares()}")
            self.agent.memory.clear_buffer()
            #
            print(f"SAC training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000 :,}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"SAC validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000 :,}")
            validation_history.append(validation_wealth - 1000000)
            if validation_wealth > max_wealth:
                saved_iter = iteration
                self.agent.save_models(round, saved_iter, self.checkpoint_dir)
            max_wealth = max(max_wealth, validation_wealth)
            if validation_history[-5:].count(max_wealth - 1000000) != 1:
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
                      f"Cumulative Return: {int(wealth) - 1000000},\tShares: {self.env.get_shares()}")
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
            self.agent.remember(observation, action, reward, observation_, done)
            self.agent.learn()
            observation = observation_
            if verbose:
                print(f"SAC testing - Date: {info.date()},\tBalance: {int(self.env.get_balance())},\t"
                      f"Cumulative Return: {int(wealth) - 1000000 :,},\tShares: {self.env.get_shares()}")
            return_history.append(wealth - 1000000)
            wealth_history.append(wealth)
        self.agent.memory.clear_buffer()

        add_curve(return_history, 'SAC')
        save_plot(self.figure_dir + f'/{self.repeat}2_testing.png',
                  title=f"Testing - {self.intervals['testing'][0].date()} to {self.intervals['testing'][1].date()}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

        returns = pd.Series(wealth_history, buy_hold_history.index).pct_change().dropna()
        stats = timeseries.perf_stats(returns)
        stats.to_csv(self.figure_dir + f'/{self.repeat}3_perf.csv')