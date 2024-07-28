import pandas as pd
import pickle 
import cvxpy as cp
from utils.problem_status import problem_status

# Load data from Excel file
input_data = pd.read_xlsx("data/data_input_dist_shifts.xlsx")
# input_data = pd.read_csv("data/data_input.csv")

from utils.battery_managamenet_preprocessing import BatteryManagementPreprocessing

E = 5000
SOC_min = 0.2
SOC_max = 0.8
max_charge_discharge = 0.1

p = input_data["Tariff"][-697-23:-23].values
d = input_data["Demand"][-697-23:-23].values

H = 697
SOC_init = 0.5

charge_discharge = cp.Variable(H)  # Battery charge/discharge
state_of_charge = cp.Variable(H + 1)  # State of charge of the battery

d = d / 1000
E = E / 1000
# Objective Function
objective = cp.Minimize(p @ (d + charge_discharge * E) )
# objective = cp.Minimize(p @ (d + charge_discharge) )

# Constraints
constraints = [
    state_of_charge[1:] == state_of_charge[:-1] + charge_discharge,
    SOC_min <= state_of_charge,
    SOC_max >= state_of_charge,
    state_of_charge[0] == SOC_init,  # Initial state of charge constraint
    cp.abs(charge_discharge) <= max_charge_discharge  # Constraint to bound charge/discharge at each instance
]

# Formulate the Problem
problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.ECOS, verbose=False)



# Check the solution status and handle accordingly
if problem.status == cp.OPTIMAL:
    # Feasible solution found
    charge_discharge_values = charge_discharge.value
    state_of_charge_values = state_of_charge.value
    objective_value = problem.value
    print(objective_value * 10)

problem_status(problem) # just a utility function to check the status of the problem (infeasible, unbounded, etc.)