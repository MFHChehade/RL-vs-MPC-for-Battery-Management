There are three folders:

1) MPC: the mpc model, with MPC_main computing the confidence interval of the cost based on 30 samples.
2) RL (for discrete action spaces): contains:
   - the environment in the environments folder
   - the different policy networks in the policies folder
   - the different RL algorithms (A2C, PPO, etc) in the agents folder
   - value estimator network and replay buffer in the learning_utils folder
   - plotting functions in rl_monitoring_utils
   - several notebooks for testing hyperparameters, testing the warm start, implementing results in general, trying out different ideas for DQN, etc
3) RL - cont action spaces: uses Stable Baselines 3 for continuous action spaces:
   - environments: contains the environment
   - rl_monitoring utils: for plotting and environment vectorization
   - data: contains the datasets
   - agents: contains the agent class
   - several notebooks to implement the different algorithms
5) old work: contains the old work using Stable Baselines 3 for discrete action spaces 
