import seaborn as sns; sns.set()
import numpy as np
import gym
from rl_monitoring_utils.return_and_advantage import calculate_returns

def REINFORCE(env, agent, gamma=0.99, epochs=100, T=1000):
    # for learning
    states = np.empty((T, env.num_envs, agent.N))
    if isinstance(env.action_space, gym.spaces.Discrete):
        # discrete action spaces only need to store a
        # scalar for each action.
        actions = np.empty((T, env.num_envs))
    else:
        # continuous action spaces need to store a
        # vector for each action.
        actions = np.empty((T, env.num_envs, agent.M))
    rewards = np.empty((T, env.num_envs))
    dones = np.empty((T, env.num_envs))

    # for plotting
    totals = []

    for epoch in range(epochs):
        s_t = env.reset()

        for t in range(T):
            a_t, _ = agent.act(s_t)
            s_t_next, r_t, d_t = env.step(a_t)

            # for learning
            states[t] = s_t
            actions[t] = a_t
            rewards[t] = r_t
            dones[t] = d_t

            s_t = s_t_next

        returns = calculate_returns(rewards, dones, gamma)
        agent.learn(states, actions, returns, update_method='PG')

        # for plotting
        # average reward = total reward/number of episodes
        totals.append(rewards.sum()/dones.sum())
        print(f'{epoch}/{epochs}:{totals[-1]}\r', end='')

    sns.lineplot(x=range(len(totals)), y=totals)

    return totals