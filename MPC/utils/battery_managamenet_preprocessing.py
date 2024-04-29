import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def BatteryManagementPreprocessing(df, horizon = 24, train_size = 30):
    """
    Preprocess the data for battery management
    - param df(pd.DataFrame): The input data
    - param horizon(int): The horizon of the MPC, default is 24
    - param train_size(int): The size of the training set, default is 30 (days)
    """

    # change each column of the data frame into a NumPy array
    data_hour = df['Hour'].values.reshape(-1,1)
    data_load = df['Demand'].values.reshape(-1,1)
    data_price = df['Tariff'].values.reshape(-1,1)
    data_weekday = df['Weekday'].values.reshape(-1,1)

    # normalize the data (except for weekday - binary values)
    sc_hour=MinMaxScaler()
    sc_load=MinMaxScaler()
    sc_price=MinMaxScaler()
    data_sc_hour=sc_hour.fit_transform(data_hour)
    data_sc_load=sc_load.fit_transform(data_load)
    data_sc_price=sc_price.fit_transform(data_price)

    # create a data frame for each column
    data_sc_hour_df = pd.DataFrame(data_sc_hour, columns = ['Scaled'])
    data_sc_load_df = pd.DataFrame(data_sc_load, columns = ['Scaled'])
    data_sc_price_df = pd.DataFrame(data_sc_price, columns = ['Scaled'])
    data_sc_weekdday_df = pd.DataFrame(data_weekday, columns = ['Scaled'])

    # create a new column for each data frame that contains the values of the previous 23 hours
    for i in range(1, horizon):
        data_sc_hour_df['shift_{}'.format(i)] = data_sc_hour_df.Scaled.shift(i)
        data_sc_load_df['shift_{}'.format(i)] = data_sc_load_df.Scaled.shift(i)
        data_sc_price_df['shift_{}'.format(i)] = data_sc_price_df.Scaled.shift(i)
        data_sc_weekdday_df['shift_{}'.format(i)] = data_sc_weekdday_df.Scaled.shift(i)

    # remove the rows with NaN values
    set_hour = data_sc_hour_df.dropna().reset_index(drop=True)
    set_load = data_sc_load_df.dropna().reset_index(drop=True)
    set_price = data_sc_price_df.dropna().reset_index(drop=True)
    set_weekday = data_sc_weekdday_df.dropna().reset_index(drop=True)

    # create the training and testing sets: training set: first month, testing set: second month
    # features: hour, weekday: real values; load, price: previous 23 values
    # target: load, price: real values
    X_train_hour=set_hour.iloc[horizon:24*(train_size + 1)].values 
    X_train_load=set_load.iloc[0:horizon*train_size].values 
    Y_train_load=set_load.iloc[horizon:horizon*(train_size + 1)].values
    X_train_price=set_price.iloc[0:horizon*train_size].values
    Y_train_price=set_price.iloc[horizon:horizon*(train_size + 1)].values
    X_train_weekday=set_weekday.iloc[horizon:horizon*(train_size + 1)].values

    X_test_hour=set_hour.iloc[horizon*(train_size + 1):len(set_hour)].values
    X_test_load=set_load.iloc[horizon*(train_size + 0):len(set_load)-horizon].values
    Y_test_load=set_load.iloc[horizon*(train_size + 1):len(set_load)].values
    X_test_price=set_price.iloc[horizon*(train_size + 0):len(set_price)-horizon].values
    Y_test_price=set_price.iloc[horizon*(train_size + 1):len(set_hour)].values
    X_test_weekday=set_price.iloc[horizon*(train_size + 1):len(set_price)].values

    # create full areas for training and testing sets
    X_train=np.dstack([X_train_hour,X_train_load,X_train_price,X_train_weekday]).astype(float)
    Y_train=np.dstack([Y_train_load,Y_train_price]).astype(float)
    X_test=np.dstack([X_test_hour,X_test_load,X_test_price,X_test_weekday]).astype(float)
    Y_test=np.dstack([Y_test_load,Y_test_price]).astype(float)

    return X_train, Y_train, X_test, Y_test, sc_load, sc_price