import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC, TD3
from sb3_contrib import TRPO, ARS, QRDQN, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_monitoring_utils.vec_monitor import VecMonitor
from rl_monitoring_utils.reward_callback import SaveOnBestTrainingRewardCallback
from rl_monitoring_utils.plotting import plot_results

class Agent:
    def __init__(self, env_id, algorithm, num_envs=8, log_dir=None):
        """
        Initialize the Agent object.

        Parameters:
        - env_id (str): Identifier for the environment.
        - algorithm (str): Identifier for the algorithm used by the agent.
        - num_envs (int, optional): Number of environments to run in parallel. Defaults to 8.
        - log_dir (str, optional): Directory to store log files. If not provided, a default directory
                                   based on the algorithm name will be created in 'results_archive'.
        """
        self.env_id = env_id
        self.algorithm = algorithm
        self.num_envs = num_envs
        self.log_dir = log_dir if log_dir else f"results_archive/{algorithm}_energy_management" # Default log directory
        os.makedirs(self.log_dir, exist_ok=True) # Create the log directory if it doesn't exist
        self.env = gym.make(env_id) # Create a single instance of the environment
        self.vec_env = self._create_vectorized_env() # Create the vectorized environment
        self.model = self._create_model() # Create the model

    ## ------------------- Initialization ------------------- ##
    
    def _create_vectorized_env(self):

        def make_env():
            return gym.make(self.env_id)

        # Create multiple instances of the environment using DummyVecEnv
        vec_env = DummyVecEnv([make_env for _ in range(self.num_envs)])
        
        # Normalize observations across the vectorized environment
        # and clip observations to avoid extreme values
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # Monitor the vectorized environment for logging purposes
        return VecMonitor(vec_env, self.log_dir)


    def _create_model(self): # Create the model based on the algorithm
        if self.algorithm == 'ppo':
            return PPO("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'trpo':
            return TRPO("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'dqn':
            return DQN("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'a2c':
            return A2C("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'ars':
            return ARS("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'qrdqn':
            return QRDQN("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'recurrentppo':
            return RecurrentPPO("MlpLstmPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'ddpg':
            return DDPG("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'sac':
            return SAC("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        elif self.algorithm == 'td3':
            return TD3("MlpPolicy", self.vec_env, verbose=1, gamma=1)
        else:
            raise ValueError("Invalid algorithm. Supported algorithms: 'ppo', 'trpo', 'dqn'")


    ## ------------------- Training ------------------- ##
        
    def train(self, total_timesteps=5e6, callback_freq=100): # Train the model
        callback = SaveOnBestTrainingRewardCallback(check_freq=callback_freq, log_dir=self.log_dir)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.model.save(f"{self.algorithm}_energy_management_model")

        print("Training Done")


    def load_model(self, model_path): # Load a pre-trained model
        self.model = self.model.load(model_path)
    
    def plot_training_curve(self): # Plot the training curve
        plot_results([self.log_dir], title="Learning Curve")

    
    ## ------------------- Testing ------------------- ##
    
    def predict(self, observation): # Make predictions using the model
        return self.model.predict(observation)

    def simulate(self, num_steps=1000): # Simulate the model
        observations = self.env.reset()
        actions = []

        for t in range(num_steps):
            action, _ = self.predict(observations)
            observations, _, _, _ = self.env.step(action[0])
            actions.append(action[0])
        
        return actions
    
    def evaluate(self, num_episodes=5): # Evaluate the model
        rewards = []

        for _ in range(num_episodes):
            obs = self.env.reset_test()
            episode_reward = 0
            i = 0
            while True:
                action, _ = self.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                i += 1

                if done:
                    break

            rewards.append(episode_reward)
        
        return - np.mean(rewards) * self.env.reward_scale

    def plot_action_distribution(self, num_steps=1000): # Plot the action distribution
        observations = self.env.reset()
        actions = []

        for t in range(num_steps):
            action = self.model.predict(observations)
            observations, _, _, _ = self.env.step(action[0])
        plt.figure(figsize=(12, 6))

        # Plotting the histogram of action distribution
        plt.hist(actions, bins=10, color='skyblue', edgecolor='black')
        plt.title('Distribution of Actions')
        plt.xlabel('Action Values')
        plt.ylabel('Frequency')
        plt.grid(True)

    ### ------------------- Getters ------------------- ### 
    def get_log_dir(self): 
        return self.log_dir
    
    def get_model(self):
        return self.model
    
    def get_env(self):
        return self.vec_env