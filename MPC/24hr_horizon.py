import pandas as pd

# Load data from Excel file
input_data = pd.read_xlsx("data/data_input_dist_shifts.xlsx")
# input_data = pd.read_csv("data/data_input.csv")

from utils.battery_managamenet_preprocessing import BatteryManagementPreprocessing

horizon = 24

# Preprocess data
X_train, Y_train, X_test, Y_test, sc_load, sc_price = BatteryManagementPreprocessing(df=input_data, horizon=horizon, train_size=int(30 * 24 / horizon))


from forecaster.LSTM_forecaster import LSTMForecaster
forecaster = LSTMForecaster(X_train, Y_train, horizon = horizon)
forecaster.train()

from mpc.MPC_battery_management import BatteryManagement
mpc = BatteryManagement(
    model=forecaster, X_test=X_test, Y_test=Y_test, horizon=horizon,
    total_time=len(X_test), X0=[0.5], E=5000, SOC_min=0.2, SOC_max=0.8,
    max_charge_discharge=0.1, sc_load=sc_load, sc_price=sc_price
)

mpc.run()
print(f"The cost of mpc is {mpc.calculate_cost()}")

print(f"The ground truth is {mpc.ground_truth()[2]}")
