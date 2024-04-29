import numpy as np
from .MPC import MPC
import cvxpy as cp
from utils.problem_status import problem_status

class BatteryManagement(MPC):

    def __init__(self, model, X_test, Y_test, horizon = 24, total_time = None, X0 = [0.5], E = 1000, SOC_min = 0.2, SOC_max = 0.8, max_charge_discharge = 0.1, sc_load = None, sc_price = None):
        """
        Initialize the BatteryManagement class
        - model: The model used to predict the future states
        - X_test (np.ndarray): The input data for the model
        - Y_test(np.ndarray): The output data for the model
        - horizon(int): The horizon of the MPC
        - total_time(int): The total time of the simulation
        - X0(np.ndarray): The initial state of the system
        - E(float): The energy capacity of the battery
        - SOC_min(float): The minimum state of charge of the battery
        - SOC_max(float): The maximum state of charge of the battery
        - max_charge_discharge(float): The maximum charge/discharge rate of the battery
        - sc_load: The MinMaxScaler object for the load data
        - sc_price: The MinMaxScaler object for the price data
        """

        super().__init__(model, X_test, Y_test, horizon, total_time, X0)
        self.E = E
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.max_charge_discharge = max_charge_discharge
        self.sc_load = sc_load
        self.sc_price = sc_price

    def optimize(self, predicted_inputs):
        """
        multi-step battery management convex optimization
        - predicted_inputs(list): two arrays of predicted inputs (p: price, d: demand)
        """
        E = self.E
        SOC_min = self.SOC_min
        SOC_max = self.SOC_max
        max_charge_discharge = self.max_charge_discharge

        p = predicted_inputs[0]
        d = predicted_inputs[1]

        H = self.horizon
        SOC_init = self.state_tracker[-1]

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

        # Solve the Problem
        problem.solve(solver=cp.SCS, verbose=False)

        # Check the solution status and handle accordingly
        if problem.status == cp.OPTIMAL:
            # Feasible solution found
            charge_discharge_values = charge_discharge.value
            state_of_charge_values = state_of_charge.value
            objective_value = problem.value
            return charge_discharge_values, state_of_charge_values, objective_value
        
        problem_status(problem) # just a utility function to check the status of the problem (infeasible, unbounded, etc.)

        return None, None, None
    
    def model_predict(self, t):

        '''
        Predict the future states using the model
        - t(int): The current time step
        '''

        input = np.reshape(self.X_test[t,:,:],(1,self.horizon,self.X_test.shape[2]))
        pd = self.model.make_prediction(input)
        
        d=self.sc_load.inverse_transform(pd[:,:,0]).T.reshape(self.horizon)
        p=self.sc_price.inverse_transform(pd[:,:,1]).T.reshape(self.horizon)
        act_load=self.sc_load.inverse_transform(np.reshape(self.Y_test[t,:,0],(self.horizon,1)))
        act_price=self.sc_price.inverse_transform(np.reshape(self.Y_test[t,:,1],(self.horizon,1)))

        p[-1] = act_price.reshape(self.horizon)[-1]
        d[-1] = act_load.reshape(self.horizon)[-1]

        p = np.flip(p)
        d = np.flip(d)
        
        
        return p[:self.horizon], d[:self.horizon]
    
    def calculate_cost(self):
        act_load = self.sc_load.inverse_transform(np.reshape(self.Y_test[:,0,0],(len(self.Y_test),1)))
        act_price = self.sc_price.inverse_transform(np.reshape(self.Y_test[:,0,1],(len(self.Y_test),1)))

        for i in range(len(self.solutions)):
            charge_discharge = self.solutions[i]
            d = act_load[i]
            p = act_price[i]
            self.cost += np.sum(p * 1e-2 * (d + charge_discharge * self.E))
        return self.cost





    