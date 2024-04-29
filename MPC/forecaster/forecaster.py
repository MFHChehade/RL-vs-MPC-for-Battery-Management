import numpy as np
from tqdm import tqdm
import inspect

class Forecaster: 

    def __init__(self, X_train, Y_train, horizon):
        """
        - param X_train(np.ndarray): The input data for the model
        - param Y_train(np.ndarray): The output data for the model
        - param horizon(int): The horizon of the MPC
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.horizon = horizon
        self.model = None

    def train(self):
        """
        Train the model
        """
        raise NotImplementedError

    def make_prediction(self, X, verbose=0):
        """
        Forecast the future states
        - param X (np.ndarray): The input data for the model
        - param verbose (int): The level of verbosity, 0 for silent, 1 for progress bar
        """
        X_reshaped = X.reshape(-1, self.horizon * X.shape[2])
        params = inspect.signature(self.model.predict).parameters
        if 'verbose' in params:
            return self.model.predict(X, verbose=verbose)
        else:
            return self.model.predict(X_reshaped)
