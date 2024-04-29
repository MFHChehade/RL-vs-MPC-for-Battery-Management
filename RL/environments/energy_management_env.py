import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class EnergyManagementEnv(gym.Env):
    def __init__(
        self,
        SOC_min: float,
        SOC_max: float,
        E: float,
        lambda_val: float,
        data_path: str,
        initial_SOC: float = None
    ):
        """
        Initialize the EnergyManagementEnv environment.

        Parameters:
        - SOC_min (float): Minimum state of charge for the battery.
        - SOC_max (float): Maximum state of charge for the battery.
        - E (float): Energy capacity of the battery.
        - lambda_val (float): Scaling factor for the penalty term in the reward function.
        - data_path (str): Path to the data file for electricity demand and prices.
        - initial_SOC (float, optional): Initial state of charge for the battery (default is None).

        """
        super(EnergyManagementEnv, self).__init__()

        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.E = E
        self.lambda_val = lambda_val

        self.data, self.scaler_demand, self.scaler_price = self.load_data(data_path)

        self.current_index = 0
        self.initial_SOC = initial_SOC
        self.bt_old = 0.0

        self.spec = gym.envs.registration.EnvSpec(
            "EnergyManagement-v0",
            entry_point="energy_management_env:EnergyManagementEnv"
        )

        self.bt_min = -0.1
        self.bt_max = 0.1

        self.reward_scale = 12500

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0, SOC_min, -np.inf, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([np.inf, SOC_max, np.inf, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        )

        self.state = self.get_initial_state()


    # --- Environment Setup ---
    def load_data(self, data_path):
        # Load data from provided CSV file using Pandas
        data = pd.read_csv(data_path)

        # Convert 'Date' column to datetime (if applicable) and extract month, day, hour
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = data['Date'].dt.day
        data['Month'] = data['Date'].dt.month

        # Extract Demand and Price columns
        demand = data['Demand'].values
        price = data['Price'].values
        day = data['Day'].values
        month = data['Month'].values
        hour = data['Hour'].values

        # Min-Max scaling for 'Demand' and 'Price' columns (if needed)
        scaler_demand = MinMaxScaler()
        scaler_price = StandardScaler()

        demand_scaled = scaler_demand.fit_transform(demand.reshape(-1, 1))
        price_scaled = scaler_price.fit_transform(price.reshape(-1, 1))

        data_with_features = np.column_stack((demand_scaled, price_scaled, hour, day, month))

        return data_with_features, scaler_demand, scaler_price


    def get_initial_state(self, x = 23):
        # x = 0  # Assuming you want to start from the first row
        self.current_index = x  # Reset time step index
        initial_row = self.data[self.current_index]  # Assuming data is a NumPy array
        initial_demand, initial_price, hour, day, month = initial_row[:5]

        initial_SOC = np.random.uniform(self.SOC_min, self.SOC_max)
        initial_month_sin = np.sin(2 * np.pi * month / 12)
        initial_month_cos = np.cos(2 * np.pi * month / 12)
        initial_day_sin = np.sin(2 * np.pi * day / 31)
        initial_day_cos = np.cos(2 * np.pi * day / 31)
        initial_hour_sin = np.sin(2 * np.pi * hour / 24)
        initial_hour_cos = np.cos(2 * np.pi * hour / 24)

        return np.array([initial_demand, initial_SOC, initial_price,
                         initial_month_sin, initial_month_cos,
                         initial_day_sin, initial_day_cos,
                         initial_hour_sin, initial_hour_cos])


    def reset(self):
        # Set the initial state based on the first time step in the dataframe
        self.state = self.get_initial_state()
        self.bt_old = 0.0  # Reset bt_old
        self.steps = 0
        return self.state


    # --- Interaction with the Environment ---
    def step(self, action):
        # Map the discrete actions to actual values: 0 -> -0.1, 1 -> 0, 2 -> 0.1
        discrete_actions = np.array([-0.1, 0, 0.1])
        bt = discrete_actions[action]

        # Extract scaled demand and price from the state
        dt_scaled, SOC_t, pt_scaled, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos = self.state

        # Inverse scaling to get original demand and price values
        dt = self.scaler_demand.inverse_transform([[dt_scaled]])[0][0]
        pt = self.scaler_price.inverse_transform([[pt_scaled]])[0][0]

        # Calculate grid energy and new state of charge
        SOC_next = SOC_t + bt
        if SOC_next < self.SOC_min:
            SOC_next = self.SOC_min
            bt = 0
        elif SOC_next> self.SOC_max:
            SOC_next = self.SOC_max
            bt = 0

        # Calculate raw reward
        raw_reward = -(pt * 1e-2 * (dt + bt * self.E))

        # Normalize reward based on reward scaling factor
        normalized_reward = raw_reward / self.reward_scale

        # Increment time step index
        self.current_index += 1
        self.steps += 1

        if self.current_index == len(self.data):
            self.current_index = 0

        # Determine if the episode is done based on the number of steps
        done = self.steps == 24 * 30
        # done = self.steps == 24 * 1 # JUST FOR TESTING DQN

        # Extract demand, price, and update state elements for the next step
        next_row = self.data[self.current_index]
        next_demand, next_price, next_hour, next_day, next_month = next_row[:5]

        # Update state
        self.state = np.array([next_demand, SOC_next, next_price,
                            np.sin(2 * np.pi * next_month / 12), np.cos(2 * np.pi * next_month / 12),
                            np.sin(2 * np.pi * next_day / 31), np.cos(2 * np.pi * next_day / 31),
                            np.sin(2 * np.pi * next_hour / 24), np.cos(2 * np.pi * next_hour / 24)])

        return self.state, normalized_reward, done, {}


    def reset_test(self):
        # Set the initial state based on the first time step in the dataframe
        self.state = self.get_initial_state(x = 768)
        self.bt_old = 0.0  # Reset bt_old
        self.steps = 0
        return self.state

    def test_agent(self, agent, gamma=1, T=697):

        '''
        - env (gym.Env): The environment to test the agent on, i.e. 1 parallel environment.s
        - agent (Agent): A trained agent.
        - gamma (float): The discount factor.
        - T (int): The number of timesteps to run the environment for in each epoch.
        '''

        # for learning
        states = np.empty((T, 1, agent.N))
        if isinstance(self.action_space, gym.spaces.Discrete):
            # discrete action spaces only need to store a
            # scalar for each action.
            actions = np.empty((T, 1))
        else:
            # continuous action spaces need to store a
            # vector for each action.
            actions = np.empty((T, 1, agent.M))
        rewards = np.empty((T, 1))
        dones = np.empty((T, 1))

        # for plotting
        totals = []

        s_t = self.reset_test()

        for t in range(T):
            # print inverse sin month st[3] and cos month st[4]
            # print inverse sin day st[5] and cos day st[6]
            # now print but organized
            a_t, _ = agent.act(s_t)
            s_t_next, r_t, d_t, _ = self.step(a_t)

            # for learning
            states[t] = s_t
            actions[t] = a_t
            rewards[t] = r_t
            dones[t] = d_t

            s_t = s_t_next

        # returns = calculate_returns(rewards, dones, gamma)

        cost = rewards.sum() * self.reward_scale

        return cost