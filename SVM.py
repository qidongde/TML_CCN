from scipy.io import loadmat
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import math

import time


def train_test_split_func(num):
    # load raw data
    filename = 'data_aggregation_final_valid'
    raw_data = loadmat(filename)

    input_final_ave = raw_data['input_final_ave']
    input_final_ave_weight = raw_data['input_final_ave_weight']
    input_final_last_merra = raw_data['input_final_last'][0][0][0].reshape(39578, 25, 1)
    input_final_last_with_local = raw_data['input_final_last'][0][0][1].reshape(39578, 25, 1)
    target_CCN = raw_data['target_CCN']
    time_traj = raw_data['time_traj']

    x_data_sum = np.concatenate((input_final_ave, input_final_ave_weight, input_final_last_merra,
                                 input_final_last_with_local), axis=2)
    x_data_chosen = x_data_sum[:, :, num]
    # x_data_chosen = input_final_ave_weight[:, :, 5]
    # x_data_chosen = input_final_last_with_local

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


def train_test_split_func2(num):
    time_traj = loadmat('data_aggregation_final_valid')['time_traj']
    target_CCN = loadmat('data_aggregation_final_valid')['target_CCN']
    # load raw data
    x_data_raw = np.load('data_aggregation_final_valid_daily.npy')
    x_data_chosen = x_data_raw[:, :, num:].reshape(39578, -1)
    # x_data_chosen = input_final_ave_weight[:, :, 5]
    # x_data_chosen = input_final_last_with_local

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


def XGB_method():
    for num in range(6):
        start_time = time.time()
        print(f'The XGBoost result of num{num + 1}:')
        x_train, x_test, y_train, y_test = train_test_split_func2(num)

        xgbr = xgb.XGBRegressor(objective='reg:squarederror')
        xgbr.fit(x_train, y_train)

        xgbr_y_pred = xgbr.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, xgbr_y_pred))
        print('test_RMSE: ', test_rmse)
        test_r2score = math.pow(pearsonr(y_test, xgbr_y_pred)[0], 2)
        print('test_r2: ', test_r2score)

        xgbr_y_pred_train = xgbr.predict(x_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, xgbr_y_pred_train))
        print('train_RMSE: ', train_rmse)
        train_r2score = math.pow(pearsonr(y_train, xgbr_y_pred_train)[0], 2)
        print('train_r2: ', train_r2score)

        end_time = time.time()
        time_consuming = end_time - start_time
        print(f'time consuming: {time_consuming:.2f}s')
        print('*' * 30)


def clf_XGB_method():
    test_result_list = []
    train_result_list = []
    for num in range(6):
        start_time = time.time()
        print(f'The XGBoost result of num{num + 1}:')
        x_train, x_test, y_train, y_test = train_test_split_func2(num)

        params = {'max_depth': [3, 6, 10],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'n_estimators': [100, 500, 600, 700, 800, 1000],
                  'colsample_bytree': [0.3, 0.7]}
        xgbr = xgb.XGBRegressor(seed=20)
        clf = GridSearchCV(estimator=xgbr,
                           param_grid=params,
                           scoring='neg_mean_squared_error',
                           verbose=1)
        clf.fit(x_train, y_train)
        print("Best parameters:", clf.best_params_)

        xgbr_y_pred = clf.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, xgbr_y_pred))
        print('test_RMSE: ', test_rmse)
        test_r2score = math.pow(pearsonr(y_test, xgbr_y_pred)[0], 2)
        print('test_r2: ', test_r2score)

        xgbr_y_pred_train = clf.predict(x_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, xgbr_y_pred_train))
        print('train_RMSE: ', train_rmse)
        train_r2score = math.pow(pearsonr(y_train, xgbr_y_pred_train)[0], 2)
        print('train_r2: ', train_r2score)

        end_time = time.time()
        time_consuming = end_time - start_time
        print(f'time consuming: {time_consuming:.2f}s')
        print('*' * 50)

        test_result_list.append(xgbr_y_pred)
        train_result_list.append(xgbr_y_pred_train)
    np.save('output_save/XGBoost_test_y_pre_daily.npy', np.array(test_result_list))
    # np.save('output_save/XGBoost_train_y_pre.npy', np.array(train_result_list))


if __name__ == '__main__':
    # XGB_method()
    clf_XGB_method()
