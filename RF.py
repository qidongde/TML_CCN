from scipy.io import loadmat
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import time


def train_test_split_func():
    # load raw data
    filename = 'data_aggregation_final_valid'
    raw_data = loadmat(filename)

    input_final_ave = raw_data['input_final_ave']
    input_final_ave_weight = raw_data['input_final_ave_weight']
    input_final_last_merra = raw_data['input_final_last'][0][0][0]
    input_final_last_iwith_local = raw_data['input_final_last'][0][0][1]
    target_CCN = raw_data['target_CCN']
    time_traj = raw_data['time_traj']

    # x_data_chosen = input_final_ave_weight[:, :, 5]
    x_data_chosen = input_final_last_merra

    test_flag = (time_traj[:, 0] == 2021) & (time_traj[:, 1] % 2 == 0)
    train_flag = (time_traj[:, 0] != 2021) | (time_traj[:, 1] % 2 != 0)

    x_train_tmp = x_data_chosen[train_flag, :]
    x_test_tmp = x_data_chosen[test_flag, :]
    y_train = target_CCN[train_flag, 0]
    y_test = target_CCN[test_flag, 0]

    standard_transfer = StandardScaler()
    x_train = standard_transfer.fit_transform(x_train_tmp)
    x_test = standard_transfer.transform(x_test_tmp)

    return x_train, x_test, y_train, y_test


def RF_method():
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split_func()

    dtr = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=9, n_jobs=-1)
    dtr.fit(x_train, y_train)
    dtr_y_pred = dtr.predict(x_test)
    rmse = mean_squared_error(y_test, dtr_y_pred)

    print('RMSE: ', rmse)
    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')


def gc_RF_method():
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split_func()

    dtr = RandomForestRegressor()
    param = {"n_estimators": [6, 12, 25, 50, 100], "max_depth": [3, 5, 7, 9, 11, 13], "random_state": [9],
             'n_jobs': [-1]}
    gc = GridSearchCV(dtr, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    gc_y_pred = gc.predict(x_test)
    rmse = mean_squared_error(y_test, gc_y_pred)

    print('RMSE: ', rmse)
    print("best_estimator_: ", gc.best_estimator_)
    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')


if __name__ == '__main__':
    RF_method()
    # gc_RF_method()
