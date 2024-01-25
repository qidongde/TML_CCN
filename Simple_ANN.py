from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from scipy.stats import pearsonr
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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

    test_flag = (time_traj[:, 0] == 2021) & (time_traj[:, 1] % 2 == 0)
    train_flag = (time_traj[:, 0] != 2021) | (time_traj[:, 1] % 2 != 0)

    x_train_tmp = x_data_chosen[train_flag, :]
    x_test_tmp = x_data_chosen[test_flag, :]
    y_train = target_CCN[train_flag, 0]
    y_test = target_CCN[test_flag, 0]

    standard_transfer = StandardScaler()
    x_train = standard_transfer.fit_transform(x_train_tmp)
    x_test = standard_transfer.transform(x_test_tmp)

    train_pairs = list(zip(x_train, y_train))
    test_pairs = list(zip(x_test, y_test))

    return train_pairs, test_pairs


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

    train_pairs = list(zip(x_train, y_train))
    test_pairs = list(zip(x_test, y_test))

    return train_pairs, test_pairs


class DataPairsDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
        self.sample_len = len(data_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index = min(max(index, 0), self.sample_len - 1)

        x = self.data_pairs[index][0]
        y = self.data_pairs[index][1]

        tensor_x = torch.tensor(x, dtype=torch.float, device=device)
        tensor_y = torch.tensor(y, dtype=torch.float, device=device)

        return tensor_x, tensor_y


class ANN_simple(nn.Module):
    def __init__(self, days):
        super(ANN_simple, self).__init__()
        self.linear1 = nn.Linear(25 * days, 60)
        self.linear2 = nn.Linear(60, 1)
        # self.linear3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        tmp1 = self.relu(self.linear1(input))
        # tmp2 = self.relu(self.linear2(tmp1))
        output = self.linear2(tmp1)
        return output.squeeze()


epochs = 40
learning_rate = 0.01
batch_size = 64


def ANN_method():
    test_result_list = []
    train_result_list = []
    for num in range(6):
        train_pairs, test_pairs = train_test_split_func2(num)
        # print(len(train_pairs))
        # print(len(test_pairs))
        train_dataset = DataPairsDataset(train_pairs)
        test_dataset = DataPairsDataset(test_pairs)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        ANN_model = ANN_simple(10 - num)
        myadam = torch.optim.Adam(ANN_model.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()
        train_rmse_loss = 0
        test_rmse_loss = 0
        train_r2score = 0
        test_r2score = 0

        for epoch_idx in range(1, epochs + 1):
            train_rmse_loss = 100
            test_rmse_loss = 100
            train_r2score = 100
            test_r2score = 100
            train_y_true = []
            train_y_pre = []
            test_y_true = []
            test_y_pre = []

            ANN_model.train()
            for train_item, (train_x, train_y) in enumerate(train_dataloader, start=1):
                train_output = ANN_model(train_x)
                train_output = train_output.to(device)
                myadam.zero_grad()

                train_loss = mse_loss(train_output, train_y)
                train_loss.backward()
                myadam.step()

                train_y_true.extend(train_y.squeeze().tolist())
                train_y_pre.extend(train_output.squeeze().tolist())
            # print(train_y_true)
            # print(train_y_pre)

            ANN_model.eval()
            with torch.no_grad():
                for test_item, (test_x, test_y) in enumerate(test_dataloader, start=1):
                    test_predict = ANN_model(test_x)
                    # print(test_predict)
                    test_y_true.extend(test_y.squeeze().tolist())
                    test_y_pre.extend(test_predict.squeeze().tolist())

            if np.sqrt(mean_squared_error(test_y_true, test_y_pre)) < test_rmse_loss:
                train_rmse_loss = np.sqrt(mean_squared_error(train_y_true, train_y_pre))
                train_r2score = math.pow(pearsonr(train_y_true, train_y_pre)[0], 2)
                test_rmse_loss = np.sqrt(mean_squared_error(test_y_true, test_y_pre))
                test_r2score = math.pow(pearsonr(test_y_true, test_y_pre)[0], 2)

                test_result_list.append(test_y_pre)
                train_result_list.append(train_y_pre)

        print(f'The ANN result of num{num + 1}:')
        print("Train RMSELoss:", train_rmse_loss)
        print("Train R2:", train_r2score)
        print("Test RMSELoss:", test_rmse_loss)
        print('Test R2', test_r2score)
        print("*" * 50)

    np.save('output_save/ANN_test_y_pre_daily.npy', np.array(test_result_list))
    # np.save('output_save/ANN_train_y_pre_daily.npy', np.array(train_result_list))


if __name__ == '__main__':
    ANN_method()
