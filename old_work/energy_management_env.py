import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class EnergyManagementEnv(gym.Env):
    def __init__(self, SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC=None):
        super(EnergyManagementEnv, self).__init__()

        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.E = E
        self.lambda_val = lambda_val
        self.data, self.scaler_demand, self.scaler_price = self.load_data(data_path)  # Use the provided data_path
        self.current_index = 0  # Initialize time step index
        self.initial_SOC = initial_SOC
        self.bt_old = 0.0  # Initialize bt_old

        self.spec = gym.envs.registration.EnvSpec("EnergyManagement-v0", entry_point="energy_management_env:EnergyManagementEnv")

        self.bt_min = - 0.1
        self.bt_max = 0.1

        self.reward_scale = 125
        # Define the action space as discrete: 0.1, 0, -0.1
        self.action_space = spaces.Discrete(3)

        # Define the state space (electricity demand, state of charge, price, month, day, hour)
        self.observation_space = spaces.Box(
            low=np.array([0, SOC_min, -np.inf, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([np.inf, SOC_max, np.inf, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        )

        # Set initial state
        self.state = self.get_initial_state()

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


    def get_initial_state(self):
        #x = np.random.randint(0, len(self.data))  # Generates a random integer from 0 to 19248 (inclusive)  # Reset time step index
        x = 0 
        # Get initial state from the first row of the data array
        self.current_index = x  # Generates a random integer from 0 to 19248 (inclusive)  # Reset time step index
        initial_row = self.data[self.current_index]  # Assuming data is a NumPy array
        initial_demand = initial_row[0]  # Assuming demand is the first column
        initial_price = initial_row[1]  # Assuming price is the second column

        initial_SOC = np.random.uniform(self.SOC_min, self.SOC_max) 
        initial_month_sin = np.sin(2 * np.pi * initial_row[4] / 12)  # Month
        initial_month_cos = np.cos(2 * np.pi * initial_row[4] / 12) 
        initial_day_sin = np.sin(2 * np.pi * initial_row[3] / 31)  # Day
        initial_day_cos = np.cos(2 * np.pi * initial_row[3] / 31)
        initial_hour_sin = np.sin(2 * np.pi * initial_row[2] / 24)  # Hour
        initial_hour_cos = np.cos(2 * np.pi * initial_row[2] / 24)

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
        gt = dt + bt
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
        normalized_reward = (raw_reward ) / self.reward_scale

        # Increment time step index
        self.current_index += 1
        self.steps += 1

        if self.current_index == len(self.data):
            self.current_index = 0

        if self.steps == 24*30:
            done = True
        else:
            done = False

 
        
        # Extract demand, price, and update state elements for the next step
        next_row = self.data[self.current_index]
        next_demand = next_row[0]  # Assuming demand is the first column
        next_price = next_row[1]  # Assuming price is the second column
        next_month_sin = np.sin(2 * np.pi * next_row[4] / 12)  # Month
        next_month_cos = np.cos(2 * np.pi * next_row[4] / 12)
        next_day_sin = np.sin(2 * np.pi * next_row[3] / 31)  # Day
        next_day_cos = np.cos(2 * np.pi * next_row[3] / 31)
        next_hour_sin = np.sin(2 * np.pi * next_row[2] / 24)  # Hour
        next_hour_cos = np.cos(2 * np.pi * next_row[2] / 24)

        # Update state
        self.state = np.array([next_demand, SOC_next, next_price,
                            next_month_sin, next_month_cos,
                            next_day_sin, next_day_cos,
                            next_hour_sin, next_hour_cos])

        return self.state, normalized_reward, done, {}



    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close(self):
        # Implement any cleanup if needed
        pass

# Function to create the environment
def energy_management_env_creator(SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC=None):
    return EnergyManagementEnv(SOC_min, SOC_max, E, lambda_val, data_path, initial_SOC)
