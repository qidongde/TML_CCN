import matplotlib.pyplot as plt
import numpy as np
import math

RF_RMSE_TEST_AVE = [0.228439834, 0.21830811, 0.224678337, 0.204293735, 0.232203809, 0.233397609]
RF_RMSE_TEST_AVE_WEIGHT = [0.225246031, 0.226330296, 0.230412941, 0.210339762, 0.225326084, 0.219078511]
RF_RMSE_TEST_LAST_MERRA = 0.242404795920464
RF_RMSE_TEST_LAST_LOCAL = 0.237768209293882
RF_RMSE_TEST_list = RF_RMSE_TEST_AVE + RF_RMSE_TEST_AVE_WEIGHT + [RF_RMSE_TEST_LAST_MERRA, RF_RMSE_TEST_LAST_LOCAL]

RF_RMSE_TRAIN_AVE = [0.240015747, 0.188258617, 0.23375034, 0.216369726, 0.251741459, 0.256616562]
RF_RMSE_TRAIN_AVE_WEIGHT = [0.241118143, 0.21412488, 0.232441696, 0.210499643, 0.228563168, 0.229391578]
RF_RMSE_TRAIN_LAST_MERRA = 0.200622082678534
RF_RMSE_TRAIN_LAST_LOCAL = 0.17274897436433

RF_R2_TEST_AVE = [0.403587115, 0.452583865, 0.422655954, 0.524477229, 0.414278202, 0.419465018]
RF_R2_TEST_AVE_WEIGHT = [0.42261767, 0.413253927, 0.397662555, 0.494903596, 0.430517206, 0.461311024]
RF_R2_TEST_LAST_MERRA = 0.324374865093662
RF_R2_TEST_LAST_LOCAL = 0.354732728522817
RF_R2_TEST_list = RF_R2_TEST_AVE + RF_R2_TEST_AVE_WEIGHT + [RF_R2_TEST_LAST_MERRA, RF_R2_TEST_LAST_LOCAL]

RF_R2_TRAIN_AVE = [0.339864493, 0.612350042, 0.371338427, 0.465808906, 0.269625364, 0.239932993]
RF_R2_TRAIN_AVE_WEIGHT = [0.332615376, 0.481046613, 0.37834963, 0.496815828, 0.399697551, 0.394499321]
RF_R2_TRAIN_LAST_MERRA = 0.574603813788552
RF_R2_TRAIN_LAST_LOCAL = 0.702201873729753

SVM_RMSE_TEST_AVE = [0.227973067, 0.222393305, 0.2248358, 0.220917759, 0.224795306, 0.229546503]
SVM_RMSE_TEST_AVE_WEIGHT = [0.23232304, 0.224480617, 0.224884252, 0.219158095, 0.216621135, 0.214663193]
SVM_RMSE_TEST_LAST_MERRA = 0.260301040782831
SVM_RMSE_TEST_LAST_LOCAL = 0.252755586638603
SVM_RMSE_TEST_list = SVM_RMSE_TEST_AVE + SVM_RMSE_TEST_AVE_WEIGHT + [SVM_RMSE_TEST_LAST_MERRA, SVM_RMSE_TEST_LAST_LOCAL]

SVM_RMSE_TRAIN_AVE = [0.243048441, 0.241225643, 0.238804329, 0.240787142, 0.245444729, 0.25240131]
SVM_RMSE_TRAIN_AVE_WEIGHT = [0.242992242, 0.240123376, 0.236404492, 0.233248753, 0.233724335, 0.234386502]
SVM_RMSE_TRAIN_LAST_MERRA = 0.258475671006767
SVM_RMSE_TRAIN_LAST_LOCAL = 0.254721277085742

SVM_R2_TEST_AVE = [0.429260233, 0.465400479, 0.441412229, 0.443328324, 0.41986003, 0.400186053]
SVM_R2_TEST_AVE_WEIGHT = [0.40353812, 0.45081476, 0.432165036, 0.450778654, 0.459683018, 0.471094595]
SVM_R2_TEST_LAST_MERRA = 0.219659541480719
SVM_R2_TEST_LAST_LOCAL = 0.265207734331628
SVM_R2_TEST_list = SVM_R2_TEST_AVE + SVM_R2_TEST_AVE_WEIGHT + [SVM_R2_TEST_LAST_MERRA, SVM_R2_TEST_LAST_LOCAL]

SVM_R2_TRAIN_AVE = [0.318371117, 0.329229222, 0.342365895, 0.335525479, 0.308517348, 0.267117588]
SVM_R2_TRAIN_AVE_WEIGHT = [0.317863821, 0.334340962, 0.355091732, 0.372363299, 0.370730063, 0.365722085]
SVM_R2_TRAIN_LAST_MERRA = 0.232571839328354
SVM_R2_TRAIN_LAST_LOCAL = 0.251732201259468

XGBoost_RMSE_TEST_AVE = [0.220597719, 0.212253719, 0.21305293, 0.200721337, 0.21349139, 0.225036215]
XGBoost_RMSE_TEST_AVE_WEIGHT = [0.22047575, 0.212123767, 0.214276116, 0.202278656, 0.209254472, 0.205622536]
XGBoost_RMSE_TEST_LAST_MERRA = 0.237839552474954
XGBoost_RMSE_TEST_LAST_LOCAL = 0.231001346935492
XGBoost_RMSE_TEST_list = XGBoost_RMSE_TEST_AVE + XGBoost_RMSE_TEST_AVE_WEIGHT + [XGBoost_RMSE_TEST_LAST_MERRA,
                                                                                 XGBoost_RMSE_TEST_LAST_LOCAL]

XGBoost_RMSE_TRAIN_AVE = [0.198282941, 0.189097633, 0.188650124, 0.190303211, 0.204509289, 0.242225791]
XGBoost_RMSE_TRAIN_AVE_WEIGHT = [0.198257334, 0.188242068, 0.192416391, 0.174146776, 0.214199504, 0.188108102]
XGBoost_RMSE_TRAIN_LAST_MERRA = 0.202402753038553
XGBoost_RMSE_TRAIN_LAST_LOCAL = 0.192270022493307

XGBoost_R2_TEST_AVE = [0.4581684, 0.494953497, 0.486119979, 0.548100222, 0.49405675, 0.437511291]
XGBoost_R2_TEST_AVE_WEIGHT = [0.458691509, 0.49604179, 0.482836577, 0.535977533, 0.505110693, 0.52689432]
XGBoost_R2_TEST_LAST_MERRA = 0.363979733747778
XGBoost_R2_TEST_LAST_LOCAL = 0.399687429758439
XGBoost_R2_TEST_list = XGBoost_R2_TEST_AVE + XGBoost_R2_TEST_AVE_WEIGHT + [XGBoost_R2_TEST_LAST_MERRA,
                                                                           XGBoost_R2_TEST_LAST_LOCAL]

XGBoost_R2_TRAIN_AVE = [0.585940846, 0.619571319, 0.616610554, 0.606103518, 0.54319443, 0.328109052]
XGBoost_R2_TRAIN_AVE_WEIGHT = [0.585907675, 0.623428788, 0.603407583, 0.668962061, 0.475759643, 0.612172765]
XGBoost_R2_TRAIN_LAST_MERRA = 0.572320223181084
XGBoost_R2_TRAIN_LAST_LOCAL = 0.611311885919627

ANN_RMSE_TEST_AVE = [0.234490566, 0.21541352, 0.219248834, 0.224454967, 0.221955454, 0.232029015]
ANN_RMSE_TEST_AVE_WEIGHT = [0.22932936, 0.211553705, 0.22172829, 0.222388406, 0.227143082, 0.219051689]
ANN_RMSE_TEST_LAST_MERRA = 0.240048878733676
ANN_RMSE_TEST_LAST_LOCAL = 0.239672666255928
ANN_RMSE_TEST_list = ANN_RMSE_TEST_AVE + ANN_RMSE_TEST_AVE_WEIGHT + [ANN_RMSE_TEST_LAST_MERRA, ANN_RMSE_TEST_LAST_LOCAL]

ANN_RMSE_TRAIN_AVE = [0.225711559, 0.224384383, 0.218544864, 0.223391514, 0.231530493, 0.241697737]
ANN_RMSE_TRAIN_AVE_WEIGHT = [0.22424123, 0.224117458, 0.218220185, 0.215765781, 0.219432267, 0.21879857]
ANN_RMSE_TRAIN_LAST_MERRA = 0.238667301532781
ANN_RMSE_TRAIN_LAST_LOCAL = 0.237268228475617

ANN_R2_TEST_AVE = [0.407896739, 0.50539417, 0.493743054, 0.471112994, 0.438930156, 0.391114638]
ANN_R2_TEST_AVE_WEIGHT = [0.399946269, 0.487989548, 0.439185597, 0.446103542, 0.475306789, 0.465788871]
ANN_R2_TEST_LAST_MERRA = 0.360528147833563
ANN_R2_TEST_LAST_LOCAL = 0.343354386467037
ANN_R2_TEST_list = ANN_R2_TEST_AVE + ANN_R2_TEST_AVE_WEIGHT + [ANN_R2_TEST_LAST_MERRA, ANN_R2_TEST_LAST_LOCAL]

ANN_R2_TRAIN_AVE = [0.411853439, 0.418377323, 0.448093781, 0.423594353, 0.380424503, 0.325064678]
ANN_R2_TRAIN_AVE_WEIGHT = [0.419284621, 0.42014764, 0.449627279, 0.462097689, 0.443602318, 0.446706774]
ANN_R2_TRAIN_LAST_MERRA = 0.34258441449993
ANN_R2_TRAIN_LAST_LOCAL = 0.349547514442645


def Test_RMSE():
    x1 = [25, 49, 73, 121, 169, 241]
    y1 = RF_RMSE_TEST_AVE
    y2 = RF_RMSE_TEST_AVE_WEIGHT
    y3 = SVM_RMSE_TEST_AVE
    y4 = SVM_RMSE_TEST_AVE_WEIGHT
    y5 = XGBoost_RMSE_TEST_AVE
    y6 = XGBoost_RMSE_TEST_AVE_WEIGHT
    y7 = ANN_RMSE_TEST_AVE
    y8 = ANN_RMSE_TEST_AVE_WEIGHT

    plt.plot(x1, y1, label='RF AVE', c='#00BFFF', marker='o')
    plt.plot(x1, y2, label='RF AVE WEIGHT', c='#00BFFF', linestyle='--', marker='x')
    plt.plot(x1, y3, label='SVM AVE', c='#FFA500', marker='o')
    plt.plot(x1, y4, label='SVM AVE WEIGHT', c='#FFA500', linestyle='--', marker='x')
    plt.plot(x1, y5, label='XGBoost AVE', c='#32CD32', marker='o')
    plt.plot(x1, y6, label='XGBoost AVE WEIGHT', c='#32CD32', linestyle='--', marker='x')
    plt.plot(x1, y7, label='ANN AVE', c='#9400D3', marker='o')
    plt.plot(x1, y8, label='ANN AVE WEIGHT', c='#9400D3', linestyle='--', marker='x')

    plt.title("TEST RMSE")
    plt.xlabel("Time step(h)")
    plt.ylabel("RMSE")
    # plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.gcf().subplots_adjust(left=0.12, right=0.65)
    plt.xticks([25, 49, 73, 121, 169, 241])
    plt.show()


def Test_R2():
    x1 = [25, 49, 73, 121, 169, 241]
    y1 = RF_R2_TEST_AVE
    y2 = RF_R2_TEST_AVE_WEIGHT
    y3 = SVM_R2_TEST_AVE
    y4 = SVM_R2_TEST_AVE_WEIGHT
    y5 = XGBoost_R2_TEST_AVE
    y6 = XGBoost_R2_TEST_AVE_WEIGHT
    y7 = ANN_R2_TEST_AVE
    y8 = ANN_R2_TEST_AVE_WEIGHT

    plt.plot(x1, y1, label='RF AVE', c='#00BFFF', marker='o')
    plt.plot(x1, y2, label='RF AVE WEIGHT', c='#00BFFF', linestyle='--', marker='x')
    plt.plot(x1, y3, label='SVM AVE', c='#FFA500', marker='o')
    plt.plot(x1, y4, label='SVM AVE WEIGHT', c='#FFA500', linestyle='--', marker='x')
    plt.plot(x1, y5, label='XGBoost AVE', c='#32CD32', marker='o')
    plt.plot(x1, y6, label='XGBoost AVE WEIGHT', c='#32CD32', linestyle='--', marker='x')
    plt.plot(x1, y7, label='ANN AVE', c='#9400D3', marker='o')
    plt.plot(x1, y8, label='ANN AVE WEIGHT', c='#9400D3', linestyle='--', marker='x')

    plt.title("TEST R2")
    plt.xlabel("Time step(h)")
    plt.ylabel("R2")
    # plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.gcf().subplots_adjust(left=0.12, right=0.65)
    plt.xticks([25, 49, 73, 121, 169, 241])
    # plt.tight_layout()
    plt.show()


test_y_true = np.load('output_save/test_y_true.npy')
train_y_true = np.load('output_save/train_y_true.npy')
RF_test_y_pre = np.load('output_save/RF_test_y_pre.npy')
RF_train_y_pre = np.load('output_save/RF_train_y_pre.npy')
SVM_test_y_pre = np.load('output_save/SVM_test_y_pre.npy')
SVM_train_y_pre = np.load('output_save/SVM_train_y_pre.npy')
XGBoost_test_y_pre = np.load('output_save/XGBoost_test_y_pre.npy')
XGBoost_train_y_pre = np.load('output_save/XGBoost_train_y_pre.npy')
ANN_test_y_pre = np.load('output_save/ANN_test_y_pre.npy')
XGBoost_daily_test_y_pre = np.load('output_save/XGBoost_test_y_pre_daily.npy')
SVM_daily_test_y_pre = np.load('output_save/SVM_test_y_pre_daily.npy')
RF_daily_test_y_pre = np.load('output_save/RF_test_y_pre_daily.npy')
ANN_daily_test_y_pre = np.load('output_save/ANN_test_y_pre_daily.npy')


# ANN_train_y_pre = np.load('output_save/ANN_train_y_pre.npy')


def y_pre_true_compare():
    model_type = 'XGBoost'
    time_step_list = ['AVE 25', 'AVE 49', 'AVE 73', 'AVE 121', 'AVE 169', 'AVE 242', 'AVE WEIGHT 25',
                      'AVE WEIGHT 49',
                      'AVE WEIGHT 73', 'AVE WEIGHT 121', 'AVE WEIGHT 169', 'AVE WEIGHT 242', 'LAST MERRA',
                      'LAST LOCAL']
    rmse_dic = {'RF': RF_RMSE_TEST_list, 'SVM': SVM_RMSE_TEST_list, 'XGBoost': XGBoost_RMSE_TEST_list,
                'ANN': ANN_RMSE_TEST_list}
    r2_dic = {'RF': RF_R2_TEST_list, 'SVM': SVM_R2_TEST_list, 'XGBoost': XGBoost_R2_TEST_list,
              'ANN': ANN_R2_TEST_list}
    y_pre_dic = {'RF': RF_test_y_pre, 'SVM': SVM_test_y_pre, 'XGBoost': XGBoost_test_y_pre,
                 'ANN': ANN_test_y_pre, 'XGBoost_daily': XGBoost_daily_test_y_pre}
    for input_num in range(12, 14):
        plt.subplot(1, 2, input_num - 11)
        x1 = np.power(10, test_y_true[:, -1])
        y1 = np.power(10, y_pre_dic[model_type][input_num, :])
        # print(y1)
        plt.title(f'{model_type} TEST {time_step_list[input_num]}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("CCN true")
        plt.ylabel("CCN pre")
        plt.scatter(x1, y1, c='#00BFFF', marker='.')
        plt.plot(x1, x1, c='black', linewidth=1.0)
        plt.axis('square')
        plt.annotate(f'RMSE = {rmse_dic[model_type][input_num]:.4f}\nR2 = {r2_dic[model_type][input_num]:.4f}',
                     xy=(200, 20))
    plt.show()


def y_pre_true_compare_daily():
    y_pre_dic = {'RF': RF_test_y_pre, 'SVM': SVM_test_y_pre, 'XGBoost': XGBoost_test_y_pre,
                 'ANN': ANN_test_y_pre, 'XGBoost_daily': XGBoost_daily_test_y_pre, 'RF_daily': RF_daily_test_y_pre,
                 'SVM_daily': SVM_daily_test_y_pre, 'ANN_daily': ANN_daily_test_y_pre}

    model_type = 'ANN_daily'

    for input_num in range(6):
        plt.subplot(1, 6, input_num + 1)
        x1 = np.power(10, test_y_true[:, -1])
        y1 = np.power(10, y_pre_dic[model_type][input_num, :])
        # print(y1)
        plt.title(f'{model_type} TEST {10 - input_num}days')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("CCN true")
        plt.ylabel("CCN pre")
        plt.scatter(x1, y1, c='#00BFFF', marker='.')
        plt.plot(x1, x1, c='black', linewidth=1.0)
        plt.axis('square')
        # plt.annotate(f'RMSE = {rmse_dic[model_type][input_num]:.4f}\nR2 = {r2_dic[model_type][input_num]:.4f}',
        #              xy=(200, 20))
    plt.show()


def y_time_ob():
    month = 12
    month_select = (test_y_true[:, 1] == month)
    # month_select = True
    x1 = test_y_true[month_select, -2]
    y1 = np.power(10, test_y_true[month_select, -1])
    y2 = np.power(10, RF_test_y_pre[3, month_select])
    y3 = np.power(10, SVM_test_y_pre[1, month_select])
    y4 = np.power(10, XGBoost_test_y_pre[3, month_select])
    y5 = np.power(10, ANN_test_y_pre[2, month_select])

    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)
    x_stick = np.linspace(x1[0], x1[-1], num=10)

    # plt.title("")
    plt.xlabel(f"2021-{month}")
    plt.ylabel("CCN concentration")
    axes.scatter(x1, y2, c='#00BFFF', marker='x', label='CCN_RF_pre')
    axes.scatter(x1, y3, c='#FFA500', marker='x', label='CCN_SVM_pre')
    axes.scatter(x1, y4, c='#32CD32', marker='x', label='CCN_XGBoost_pre')
    axes.scatter(x1, y5, c='#9400D3', marker='x', label='CCN_ANN_pre')
    axes.scatter(x1, y1, c='black', marker='.', label='CCN_true')
    axes.set_aspect(aspect=6 / 1)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.gcf().subplots_adjust(left=0.12, right=0.75)
    plt.xticks(x_stick, ['1', '4', '7', '10', '13', '16', '19', '22', '25', '28'])
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    # Test_RMSE()
    # Test_R2()
    # y_pre_true_compare()
    # y_time_ob()
    y_pre_true_compare_daily()
