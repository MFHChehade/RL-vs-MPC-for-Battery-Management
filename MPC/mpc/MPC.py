import numpy as np
from tqdm import tqdm

class MPC:

    def __init__(self, model, X_test, Y_test, horizon, total_time, X0):
        """
        Initialize the MPC class
        - param model: The model used to predict the future states
        - param X_test (np.ndarray): The input data for the model
        - param Y_test(np.ndarray): The output data for the model
        - param horizon(int): The horizon of the MPC
        - param total_time(int): The total time of the simulation   
        - param X0(np.ndarray): The initial state of the system
        """

        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.horizon = horizon
        if total_time == None: 
            total_time = len(X_test)
        self.total_time = total_time
        self.X0 = X0
        self.cost = 0 # Initialize the cost of the MPC
        self.solutions = []  # List to store optimal control actions
        self.state_tracker = [self.X0]  # List to store states, if necessary


    def optimize(self):
        """
        multi-step optimization
        """
        raise NotImplementedError
    
    
    def run(self):
        """
        Implement the MPC control loop.
        """
        for t in tqdm(range(self.total_time), desc="Solving Control Problem"):
            H = min(self.total_time - t, self.horizon)  # Calculate the time window size
            predicted_inputs = self.model_predict(t)

            # Call the optimizer function
            control_actions, states, objective_value = self.optimize(predicted_inputs)

            self.solutions.append(control_actions[0])
            self.state_tracker.append(states[1])
            print(states)

        return self.solutions, self.state_tracker
    
    def model_predict(self):
        """
        Predict the future states using the model
        - param X(np.ndarray): The input data for the model
        """
        raise NotImplementedError
    
    def calculate_cost(self):
        '''
        Finds the cost of the MPC
        '''
        raise NotImplementedError
        

        