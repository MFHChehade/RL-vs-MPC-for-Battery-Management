# Import necessary libraries
from environments.energy_management_env import EnergyManagementEnv
from rl_monitoring_utils.plotting import plot_results
from agents.agent import Agent
from environments.env_registration import register_env
from agents.agent import Agent

# Register the custom environment
env_params = {
    'SOC_min': 0.2,
    'SOC_max': 0.8,
    'E': 5000,
    'lambda_val': 0.1,
    'data_path': 'data/Data_input.csv',
    'initial_SOC': 0.5  # Set to None if not using an initial_SOC
}
register_env('EnergyManagement-v0', 'environments.env_registration:environment_creator', {'environment_class': EnergyManagementEnv, **env_params})

# Define environment ID
env_id = 'EnergyManagement-v0'

# Define total timesteps and number of environments for training
total_timesteps = 5e6
num_envs = 8

# Create a dictionary to store agents with their names
agents = {
    'ppo_agent': Agent(env_id, 'ppo', num_envs=num_envs),
    'dqn': Agent(env_id, 'trpo', num_envs=num_envs),
}

# Train agents
for agent_name, agent in agents.items():
    agent.train(total_timesteps=total_timesteps, num_runs=5)
    reward = agent.evaluate()
    print(f'{agent_name} reward: {reward}')
    

# Plot results for all agents
#log_dirs = [agent.get_log_dir() for agent in agents.values()]
# agents['trpo_agent'].plot_training_curve()

# :param log_folders_by_alg: (dict) Dictionary with algorithm names as keys and lists of log folder paths as values.
# each key is name and each value is log_dir_runs (list)

log_folders_by_alg = {agent_name: agent.log_dir_runs for agent_name, agent in agents.items()}
plot_results(log_folders_by_alg, title="Learning Curves", window=1000)
import matplotlib.pyplot as plt
def plot_action_distribution(agent, num_steps=697, title = None): # Plot the action distribution
    observations = agent.env.reset_test()
    actions = []
    rewards = []
    print(agent.env.current_index)

    for t in range(num_steps):
        action = agent.model.predict(observations)
        observations, reward, _, _ = agent.env.step(action[0])

        actions.append(action[0])
        rewards.append(reward)

    plt.figure(figsize=(12, 6))

    # Plotting the histogram of action distribution
    plt.hist(actions, bins=5)
    plt.title('Distribution of Actions for ' + title + ' Agent')
    plt.xlabel('Action Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    return actions, rewards


for name, agent in agents.items():
    actions, rewards = plot_action_distribution(agent, title=name)
    plt.show()
import pandas as pd

# Collecting costs
costs = {name: [] for name, agent in agents.items()}

for name, agent in agents.items():
    for _ in range(5):
        cost = agent.evaluate()
        costs[name].append(cost)

# Creating a DataFrame for easier plotting
df = pd.DataFrame(costs)

import matplotlib.pyplot as plt

# Creating the box plots
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title('Cost Evaluation of Agents')
plt.ylabel('Cost')
plt.xlabel('Agent')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

costs = {name: [] for name, agent in agents.items()}

for name, agent in agents.items():
    for _ in range(5):
        cost = agent.evaluate()
        costs[name].append(cost)

# Creating a DataFrame for easier plotting and analysis
df = pd.DataFrame(costs)

# Calculate the mean and the standard error of the mean
means = df.mean()
sems = df.sem()

# Determine the t-critical value for 95% CI
t_critical = stats.t.ppf(q = 0.975, df = 4)  # q is the tail probability, df is degrees of freedom (n-1)

# Calculate the margin of error
margins = t_critical * sems

# Create confidence intervals
confidence_intervals = pd.DataFrame({
    'Lower Bound': means - margins,
    'Upper Bound': means + margins
}, index=means.index)

# Print confidence intervals
print(confidence_intervals)

