import pandas as pd
import numpy as np
from scipy import stats

# Assuming you have the following files and functions properly set up:
# - battery_management_preprocessing.py with class BatteryManagementPreprocessing
# - LSTM_forecaster.py with class LSTMForecaster
# - MPC_battery_management.py with class BatteryManagement

# Load data from Excel file
# input_data = pd.read_excel("data/data_input.csv")
input_data = pd.read_excel("data/data_input_dist_shift.xlsx")

# Import classes from your files
from utils.battery_managamenet_preprocessing import BatteryManagementPreprocessing
from forecaster.LSTM_forecaster import LSTMForecaster
from mpc.MPC_battery_management import BatteryManagement

# Function to perform a single run of the experiment
def perform_experiment(horizon = 24):
    # Preprocess data
    X_train, Y_train, X_test, Y_test, sc_load, sc_price = BatteryManagementPreprocessing(df=input_data, horizon=horizon, train_size=30 * 24 / horizon)
    
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

# Run the experiment multiple times
results = [perform_experiment() for _ in range(5)]

# Calculate average cost and 95% confidence interval
average_cost = np.mean(results)
confidence_interval_95 = stats.t.interval(0.95, len(results)-1, loc=np.mean(results), scale=stats.sem(results))
confidence_interval_99 = stats.t.interval(0.99, len(results)-1, loc=np.mean(results), scale=stats.sem(results))
confidence_interval_995 = stats.t.interval(0.995, len(results)-1, loc=np.mean(results), scale=stats.sem(results))

print(f"Average MPC cost is ${average_cost:.2f}.")
print(f"99% confidence interval for the cost: {confidence_interval_99}.")
