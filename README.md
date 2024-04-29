There are three folders:

1) MPC: the mpc model, with MPC_main computing the confidence interval of the cost based on 30 samples.
2) RL: contains:
   - the environment in the environments folder
   -  the different policy networks in the policies folder
   -  the different RL algorithms (A2C, PPO, etc) in the agents folder
   -  value estimator network and replay buffer in the learning_utils folder
   -  plotting functions in rl_monitoring_utils
   -  several notebooks for testing hyperparameters, testing the warm start, implementing results in general, trying out different ideas for DQN, etc
3) old work: contains the old work using Stable Baselines 3
