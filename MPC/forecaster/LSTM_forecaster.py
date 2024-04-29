import numpy as np
from .forecaster import Forecaster
import os
from sklearn.preprocessing import MinMaxScaler
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from keras.models import Model
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
import keras.backend as K

class LSTMForecaster(Forecaster):
    def __init__(self, X_train, Y_train, horizon):
        """
        - param X_train(np.ndarray): The input data for the model
        - param Y_train(np.ndarray): The output data for the model
        - param horizon(int): The horizon of the MPC
        """
        super().__init__(X_train, Y_train, horizon)
        self.horizon = horizon


        K.clear_session() # Clear previous models from memory

        xInput = Input(batch_shape=(None, self.horizon, X_train.shape[2])) # 4 features
        xLstm = LSTM(self.horizon, return_sequences=True)(xInput) # 24 LSTM units
        xOutput = TimeDistributed(Dense(2))(xLstm) # 2 outputs: load and price

        self.model = Model(xInput, xOutput) # Create the model
        self.model.compile(loss='mean_squared_error', optimizer='adam') # Compile the model

        self.model.summary() # Display the model summary

    def train(self, epochs = 100, patience = 1, verbose = 1):
        early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=verbose) # Early stopping
        self.history=self.model.fit(self.X_train,self.Y_train, epochs=100, verbose=verbose) # Fit the model

    