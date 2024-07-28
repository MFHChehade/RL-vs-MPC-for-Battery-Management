import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils.battery_managamenet_preprocessing import BatteryManagementPreprocessing
from forecaster.LSTM_forecaster import LSTMForecaster
from mpc.MPC_battery_management import BatteryManagement

# Load data from Excel file
input_data = pd.read_xlsx("data/data_input_dist_shifts.xlsx")
# input_data = pd.read_csv("data/data_input.csv")


# Function to perform a single run of the experiment
def perform_experiment(horizon = 24, data = None):
    # Preprocess data
    X_train, Y_train, X_test, Y_test, sc_load, sc_price = BatteryManagementPreprocessing(df=data, horizon=horizon, train_size=int(30 * 24 / horizon))
    
    # Setup and train the LSTM forecaster
    forecaster = LSTMForecaster(X_train, Y_train, horizon = horizon)
    forecaster.train()

    # Setup and run the MPC
    mpc = BatteryManagement(
        model=forecaster, X_test=X_test, Y_test=Y_test, horizon=horizon,
        total_time=len(X_test), X0=[0.5], E=5000, SOC_min=0.2, SOC_max=0.8,
        max_charge_discharge=0.1, sc_load=sc_load, sc_price=sc_price
    )
    mpc.run()

    # Return the cost
    return mpc.calculate_cost()
# Define horizons
horizons = [1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336]
# horizons = [1, 2]
results = []

# Run the experiments for each horizon
for horizon in horizons:
    costs = [perform_experiment(horizon, input_data) for _ in range(30)]
    average_cost = np.mean(costs)
    confidence_interval_95 = stats.t.interval(0.95, len(costs)-1, loc=average_cost, scale=stats.sem(costs))
    results.append((average_cost, np.std(costs), confidence_interval_95))

# Plotting the results
avg_costs, std_devs, conf_intervals = zip(*results)
plt.figure(figsize=(10, 5))
plt.errorbar(horizons, avg_costs, yerr=std_devs, fmt='-o', ecolor='red', capsize=5, label='Average Cost ±1 SD')
plt.fill_between(horizons, [ci[0] for ci in conf_intervals], [ci[1] for ci in conf_intervals], color='gray', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Horizon Length (hours)')
plt.ylabel('Average MPC Cost ($)')
plt.title('Variation of Mean MPC Cost as a Function of Horizon Length')
plt.legend()
plt.grid(True)
plt.show()

