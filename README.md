# ActorCriticRL_PortfolioManagement

## [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) ([Deep Deterministic Policy Gradient](https://keras.io/examples/rl/ddpg_pendulum/))

Learns: Q-function and a policy.

It uses off-policy data and the Bellman equation to learn the Q-function (Q-learning), and uses the Q-function to learn the policy. To make DDPG policies explore better, we add noise to their actions at training time.

### Q-learning: 

$$ a^{\star}(s) = arg \ \underset{a}{max} \ Q^{\star}(s,a) $$

When there are a finite number of discrete actions, the max poses no problem, because we can just compute the Q-values for each action separately and directly compare them. This also immediately gives us the action which maximizes the Q-value. But when the action space is continuous, we can’t exhaustively evaluate the space. 

Because the action space is continuous, the function $Q^*(s,a)$ is presumed to be differentiable with respect to the action argument. This allows us to set up an efficient, gradient-based learning rule for a policy $\mu(s)$ which exploits that fact. Then, instead of running an expensive optimization subroutine each time we wish to compute $\max_a Q(s,a)$, we can approximate it with $\max_a Q(s,a) \approx Q(s,\mu(s))$.

First, let's recap the Bellman equation describing the optimal action-value function, $Q^*(s,a)$. It's given by

$$
    Q^{\star}(s,a) = \underset{s' \sim P}{{\mathrm E}}\left[r(s,a) + \gamma \max_{a'} Q^{\star}(s', a')\right]
    $$

where $s' \sim P$ is shorthand for saying that the next state, $s'$, is sampled by the environment from a distribution $P(\cdot| s,a)$. 

This Bellman equation is the starting point for learning an approximator to $Q^*(s,a)$. Suppose the approximator is a neural network $Q_{\phi}(s,a)$, with parameters $\phi$, and that we have collected a set $\tau$ of transitions $(s,a,r,s',\tau)$ (where $d$ indicates whether state $s'$ is terminal). We can set up a **mean-squared Bellman error (MSBE)** function, which tells us roughly how closely $Q_{\phi}$ comes to satisfying the Bellman equation:

$$
    L(\phi, \tau) = \underset{(s,a,r,s',d) \sim \tau}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
        \right]
$$

Here, in evaluating $(1-d)$, we've used a Python convention of evaluating ``True`` to 1 and ``False`` to 0.

Q-learning algorithms for function approximators, such as [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (and all its variants) and DDPG, are largely based on minimizing this MSBE loss function. Tricks used:

1. **Replay Buffers**. All standard algorithms for training a deep neural network to approximate $Q^{\star}(s,a)$ make use of an experience replay buffer. This is the set $\tau$ of previous experiences. In order for the algorithm to have stable behavior, the replay buffer should be large enough to contain a wide range of experiences.
2. **Target Networks**. When we minimize the MSBE loss, we are trying to make the Q-function be more like this **target**: 

$$r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a')$$ 

Problematically, the target depends on the same parameters that we are trying to train: $\phi$. This makes MSBE minimization unstable. The solution is to use a set of parameters which comes close to $\phi$, but with a time delay—that is to say, a second network, called the target network, which lags the first. In DDPG-style algorithms, the target network is updated once per main network update by polyak averaging: 

$$ \phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1 - \rho) \phi $$

3. **Calculating the Max Over Actions in the Target**. Use a **target policy network** to compute an action which approximately maximizes $Q_{\phi_{\text{targ}}}$.

Q-learning in DDPG is performed by minimizing the following MSBE loss with stochastic gradient descent:

$$
    L(\phi, \tau) = \underset{(s,a,r,s',d) \sim \tau}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
        \right]
$$

where $\mu_{\theta_{\text{targ}}}$ is the target policy.


### Policy Learning

We want to learn a deterministic policy $\mu_{\theta}(s)$ which gives the action that maximizes $Q_{\phi}(s,a)$. Because the action space is continuous, and we assume the Q-function is differentiable with respect to action, we can just perform gradient ascent (with respect to policy parameters only) to solve

$$
    \max_{\theta} \underset{s \sim \tau}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].
$$

Note that the Q-function parameters are treated as constants here.

### Pseudocode

![alt text](https://github.com/manuzrpEd/PortfolioManagement-ActorCriticRL/blob/main/imgs/DDPG.svg)


## [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) (Proximal Policy Optimization)

PPO is an on-policy algorithm. It trains a stochastic policy in an on-policy way. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.

PPO-clip updates policies via

$$
    \theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[
        L(s,a,\theta_k, \theta)\right],     
$$

typically taking multiple steps of (usually minibatch) SGD to maximize the objective. Here $L$ is given by

$$
    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), 
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)
    \right),  
$$

in which $\epsilon$ is a (small) hyperparameter which roughly says how far away the new policy is allowed to go from the old. A considerably simplified version is as follows:

$$
    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), 
    g(\epsilon, A^{\pi_{\theta_k}}(s,a))
    \right),  
$$

where

$$
  g(\epsilon, A) = 
     \begin{cases}
       (1 + \epsilon) A & A \geq 0\\
       (1 - \epsilon) A & A < 0 \\ 
     \end{cases}
$$  

is called the *advantage*.

### Pseudocode

![alt text](https://github.com/manuzrpEd/PortfolioManagement-ActorCriticRL/blob/main/imgs/PPO.svg)

## [A2C](https://adventuresinmachinelearning.com/a2c-advantage-actor-critic-tensorflow-2/) (Advantage Actor-Critic)

* <ins>Advantage Actor</ins>: the part of the neural network that is used to determine the actions of the agent. The "advantage" expresses the relative benefit of taking a certain action $a_t$ from a certain state $s_t$, $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$
* <ins>Q-value</ins>: is the expected future rewards of taking action $a_t$ from state $s_t$. The Q-value is the expected value of taking a certain action from the current state: $Q(s_t, a_t) = \mathbb{E} \big[r_{t+1} + \gamma V(s_{t+1}) \big]$.
* <ins>Value-function</ins>: the expected value of the agent being in that state and operating under a certain action policy $\pi$. It can be expressed as: $V^{\pi}(s) = \mathbb{E} \big[\displaystyle\sum_{i=1}^{T} r_{i} \big]$. Here $\mathbb{E}$ is the expectation operator, and the value $V^{\pi}(s)$ can be read as the expected value of future discounted rewards that will be gathered by the agent, operating under a certain action policy $\pi$. $V$ is the expected value of simply being in the current state, under a certain action policy.

Q-values can be a source of high variance in the training process, and it is much better to use the normalized or baseline Q-values i.e. the *advantage*, in training.

We use the following gradient function to train the neural network:

$$
\nabla_{\theta} J(\theta) \sim \big( \displaystyle\sum_{t=0}^{T} log P_{\pi_{\theta}}(s_t, a_t) \big) A(s_t, a_t).
$$

The *advantage* can be estimated as: $A(s_t, a_t) = r_{t+1} + \gamma V(s_{t+1}) - V(s_{t})$.

### A2C loss functions


1. **Critic loss**: We can compare the predicted $V(s)$ at each state in the game, and the actual sampled discounted rewards that were gathered, and the difference between the two is the Critic loss.
2. **Actor loss**: loss function need that pertains to the training of the Actor (i.e. the action policy).
3. **Entropy loss**. $E = - \displaystyle\sum p(x) log(p(x))$. By subtracting the entropy calculation from the total loss (or giving the entropy loss a negative sign), it encourages more randomness and therefore more exploration.

The total loss function for the A2C algorithm is: ``Loss = Actor Loss + Critic Loss * CRITIC_WEIGHT – Entropy Loss * ENTROPY_WEIGHT``

### Pseudocode

![alt text](https://github.com/manuzrpEd/PortfolioManagement-ActorCriticRL/blob/main/imgs/A2C.png)

## [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) (Soft Actor-Critic)

Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way. SAC concurrently learns a policy $\pi_{\theta}$ and two Q-functions $Q_{\phi_1}, Q_{\phi_2}$. A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum.

In entropy-regularized reinforcement learning, the agent gets a bonus reward at each time step proportional to the entropy of the policy at that timestep. This changes the RL problem to:

$$
\pi^* = \arg \max_{\pi} \underset{\tau \sim \pi}{E}{ \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot|s_t)\right) \bigg)},
$$

where the entropy $H$ of $x$ is computed from its distribution $P$ according to $H(P) = \underset{x \sim P}{E}{-\log P(x)}$. $V^{\pi}$ is changed to include the entropy bonuses from every timestep:  

$$
V^{\pi}(s) = \underset{\tau \sim \pi}{E}{ \left. \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot|s_t)\right) \bigg) \right| s_0 = s}
$$

$Q^{\pi}$ is changed to include the entropy bonuses from every timestep except the first:

$$
Q^{\pi}(s,a) = \underset{\tau \sim \pi}{E}{ \left. \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) + \alpha \sum_{t=1}^{\infty} \gamma^t H\left(\pi(\cdot|s_t)\right)\right| s_0 = s, a_0 = a}
$$

With these definitions, $V^{\pi}$ and $Q^{\pi}$ are connected by:

$$
V^{\pi}(s) = \underset{a \sim \pi}{E}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right)
$$

and the Bellman equation for $Q^{\pi}$ is

$$
\begin{align*}
Q^{\pi}(s,a) &= \underset{s' \sim P \\ a' \sim \pi}{E}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha H\left(\pi(\cdot|s')\right) \right)} \\ 
&= \underset{s' \sim P}{E}{R(s,a,s') + \gamma V^{\pi}(s')}.
\end{align*}
$$

### Loss functions

The loss functions for the Q-networks in SAC are:

$$
L(\phi_i, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[ \Bigg( Q_{\phi_i}(s,a) - y(r,s',d) \Bigg)^2 \right],
$$

where the target is given by

$$
y(r, s', d) = r + \gamma (1 - d) \left( \min_{j=1,2} Q_{\phi_{\text{targ},j}}(s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s') \right), \tilde{a}' \sim \pi_{\theta}(\cdot|s').
$$

The policy is thus optimized according to

$$
\max_{\theta} \underset{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{E}{\min_{j=1,2} Q_{\phi_j}(s,\tilde{a_{\theta}}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a_{\theta}}(s,\xi)|s)},
$$


### Pseudocode

![alt text](https://github.com/manuzrpEd/PortfolioManagement-ActorCriticRL/blob/main/imgs/SAC.svg)
