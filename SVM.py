import numpy
from scipy.io import loadmat
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import time


def train_test_split_func():
    # load raw data
    filename = 'data_aggregation_final_valid'
    raw_data = loadmat(filename)

    input_final_ave = raw_data['input_final_ave']
    input_final_ave_weight = raw_data['input_final_ave_weight']
    input_final_last_merra = raw_data['input_final_last'][0][0][0]
    input_final_last_with_local = raw_data['input_final_last'][0][0][1]
    target_CCN = raw_data['target_CCN']
    time_traj = raw_data['time_traj']

    # x_data_chosen = input_final_ave[:, :, 5]
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


def SVM_method():
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split_func()

    svr = LinearSVR(epsilon=0.05, random_state=9, max_iter=20000)
    svr.fit(x_train, y_train)

    svr_y_pred = svr.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, svr_y_pred))
    print('test_RMSE: ', test_rmse)
    test_r2score = r2_score(y_test, svr_y_pred)
    print('test_r2: ', test_r2score)

    svr_y_pred_train = svr.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, svr_y_pred_train))
    print('train_RMSE: ', train_rmse)
    train_r2score = r2_score(y_train, svr_y_pred_train)
    print('train_r2: ', train_r2score)

    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')


def gc_SVM_method():
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split_func()

    dtr = LinearSVR()
    param = {"epsilon": [0.005, 0.01, 0.02, 0.05, 0.1], 'random_state': [9]}
    gc = GridSearchCV(dtr, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("best_estimator_: ", gc.best_estimator_)

    gc_y_pred = gc.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, gc_y_pred))
    print('test_RMSE: ', test_rmse)
    test_r2score = r2_score(y_test, gc_y_pred)
    print('test_r2: ', test_r2score)

    gc_y_pred_train = gc.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, gc_y_pred_train))
    print('train_RMSE: ', train_rmse)
    train_r2score = r2_score(y_train, gc_y_pred_train)
    print('train_r2: ', train_r2score)

    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'time consuming: {time_consuming:.2f}s')


if __name__ == '__main__':
    # SVM_method()
    gc_SVM_method()
