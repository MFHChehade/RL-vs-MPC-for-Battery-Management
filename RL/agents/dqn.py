import torch
import numpy as np
import copy
import seaborn as sns
import pandas as pd


class DeepQLearner:
    def __init__(self, env,
                 alpha=0.001, gamma=0.95,
                 epsilon_i=1.0, epsilon_f=0.00, n_epsilon=0.1, clip_grad=1.0):
        '''
        env (VectorizedEnvWrapper): the vectorized gym.Env
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon_i (float): initial value for epsilon
        epsilon_f (float): final value for epsilon
        n_epsilon (float): proportion of timesteps over which to
                           decay epsilon from epsilon_i to
                           epsilon_f
        '''

        self.num_envs = env.num_envs
        self.M = env.action_space.n  # number of actions
        self.N = env.observation_space.shape[0] # dimensionality of state space

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.n_epsilon = n_epsilon
        self.epsilon = epsilon_i
        self.gamma = gamma
        self.clip_grad = clip_grad  # gradient clipping value

        self.Q = torch.nn.Sequential(
            torch.nn.Linear(self.N, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.M)
        ).double()

        self.Q_ = copy.deepcopy(self.Q)

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)

    def synchronize(self):
        '''
        Used to make the parameters of Q_ match with Q.
        '''
        self.Q_.load_state_dict(self.Q.state_dict())

    def act(self, s_t):
        '''
        Epsilon-greedy policy.
        s_t (np.ndarray): the current state.
        '''
        s_t = torch.as_tensor(s_t).double()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.M, size=self.num_envs)
        else:
            with torch.no_grad():
                return np.argmax(self.Q(s_t).numpy(), axis=1)

    def decay_epsilon(self, n):
        '''
        Epsilon decay.
        n (int): proportion of training complete
        '''
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n/self.n_epsilon)*(self.epsilon_i - self.epsilon_f))

    def update(self, s_t, a_t, r_t, s_t_next, d_t):
        '''
        Learning step.
        s_t (np.ndarray double): state
        a_t (np.ndarray int): action
        r_t (np.ndarray double): reward
        d_t (np.ndarray float): done flag
        s_t_next (np.ndarray double): next state
        '''

        # make sure everything is torch.Tensor and type-compatible with Q
        s_t = torch.as_tensor(s_t).double()
        a_t = torch.as_tensor(a_t).long()
        r_t = torch.as_tensor(r_t).double()
        s_t_next = torch.as_tensor(s_t_next).double()
        d_t = torch.as_tensor(d_t).double()

        # we don't want gradients when calculating the target y
        with torch.no_grad():
            # taking 0th element because torch.max returns both maximum
            # and argmax
            Q_next = torch.max(self.Q_(s_t_next), dim=1)[0]
            target = r_t + (1-d_t)*self.gamma*Q_next

        # use advanced indexing on the return to get the predicted
        # Q values corresponding to the actions chosen in each environment.
        Q_pred = self.Q(s_t)[range(self.num_envs), a_t]
        loss = torch.mean((target - Q_pred)**2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), self.clip_grad)
        self.optimizer.step()

def DQN(env, agent, replay_buffer, T=20000, n_theta=100):
    '''
    env (VectorizedEnvWrapper): vectorized gym.Env
    agent (DeepQLearner)
    buffer (ReplayBuffer)
    T (int): total number of training timesteps
    batch_size: number of
    '''

    # for plotting
    returns = []
    episode_rewards = 0

    s_t = env.reset()
    for t in range(T):
        # synchronize Q and Q_
        if t%n_theta == 0:
            agent.synchronize()

        a_t = agent.act(s_t)
        s_t_next, r_t, d_t = env.step(a_t)

        # store data into replay buffer
        replay_buffer.remember(s_t, a_t, r_t, s_t_next, d_t)
        s_t = s_t_next

        # learn by sampling from replay buffer
        for batch in replay_buffer.sample():
            agent.update(*batch)

        # for plotting
        episode_rewards += r_t
        for i in range(env.num_envs):
            if d_t[i]:
                returns.append(episode_rewards[i])
                episode_rewards[i] = 0
                
                print(f'{t}/{T}:{returns[-1]}\r', end='')

        # epsilon decay
        agent.decay_epsilon(t/T)

    window = 300
    sns.lineplot(
        data=pd.DataFrame(returns).rolling(window=window).mean()[window-1::window]
    )
    return returns, agent