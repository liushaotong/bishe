"""
预测模型在不同任务上的可迁移性
"""
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def plot_one(pre_y_list, y):
    pre_y_list = pre_y_list * 100
    y = y * 100
    plt.plot(pre_y_list, y, 'o', markersize=1.0)  # 画出每条预测结果线

    plt.title('regression result for transferability')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real value')  # y轴标题
    plt.xlabel('predicted value')
    plt.savefig('transfer.jpg')  # 保存图片
    plt.show()  # 展示图像
    return


def filter(data_train):
    df = data_train[['complexity.dist_spec_init', 'complexity.fro_dist', 'complexity.fro_over_spec',
                     'complexity.log_sum_of_fro', 'complexity.log_sum_of_spec', 'complexity.param_norm',
                     'complexity.pacbayes_flatness', 'complexity.pacbayes_init', 'complexity.pacbayes_mag_flatness',
                     'complexity.pacbayes_orig', 'complexity.inverse_margin', 'complexity.log_sum_of_fro_over_margin',
                     'complexity.log_sum_of_spec_over_margin', ]]
    df.to_csv('./svhn_filter.csv')


def main():
    data_train = pd.read_csv('./nin.cifar10_svhn.csv')
    data_train_cifar10 = data_train[np.logical_and(data_train['hp.dataset'] == 'cifar10', data_train['is.converged'] == True,
                                                data_train['is.high_train_accuracy'] == True)]
    # filter(data_train_cifar10)
    data_train_svhn = data_train[np.logical_and(data_train['hp.dataset']=='svhn', data_train['is.converged']==True,
                                                data_train['is.high_train_accuracy'] == True)]
    # filter(data_train_svhn)
    data_train_target = data_train_cifar10.loc[:, ['gen.gap']]
    y_train = data_train_target.values.flatten()
    data_train_x = pd.read_csv('./cifar10_filter.csv')  # 超参数设置
    X_train = data_train_x.values[:, 1:]


    data_test_target = data_train_svhn.loc[:, ['gen.gap']]
    y_test = data_test_target.values.flatten()
    data_test_x = pd.read_csv('./svhn_filter.csv')
    X_test = data_test_x.values[:, 1:]


    # 神经网络回归
    torch.manual_seed(1)
    train_features = torch.from_numpy(X_train.astype(np.float32)).cuda()
    train_labels = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).cuda()
    test_features = torch.from_numpy(X_test.astype(np.float32)).cuda()
    test_labels = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1).cuda()
    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)
    # 定义迭代器
    train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(dataset=test_set, batch_size=64, shuffle=False)
    # 构建网络结构
    class Net(nn.Module):  # 继承 torch 的 Module
        def __init__(self, n_feature, n_output):
            super(Net, self).__init__()  # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.layer1 = nn.Linear(n_feature, 128).cuda()  #
            self.layer2 = nn.Linear(128, 64).cuda()  #
            self.layer3 = nn.Linear(64, n_output).cuda()

        def forward(self, x):  #
            x = self.layer1(x)
            x = torch.relu(x)  #
            x = self.layer2(x)
            x = torch.relu(x)  #
            x = self.layer3(x)
            return x
    net = Net(13, 1).cuda()  # 输入维度！！
    # Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 均方损失函数
    criterion = torch.nn.MSELoss()
    losses = []
    eval_losses = []
    for i in range(100):
        train_loss = 0
        net.train()
        for tdata,tlabel in train_data:
            y_ = net(tdata)
            loss = criterion(y_, tlabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
        losses.append(train_loss / len(train_data))
        # 验证
        eval_loss = 0
        net.eval()  # 可加可不加
        for edata, elabel in test_data:
            y_ = net(edata)
            loss = criterion(y_, elabel)
            eval_loss = eval_loss + loss.item()
        eval_losses.append(eval_loss / len(test_data))
        print('epoch: {}, trainloss: {}, evalloss: {}'.format(i, train_loss / len(train_data),
                                                              eval_loss / len(test_data)))
    y_pred = net(test_features).squeeze().detach().cpu().numpy()
    all_features = torch.from_numpy(X_test.astype(np.float32)).cuda()
    pre_y_list = net(all_features).squeeze().detach().cpu().numpy()
    plot_one(pre_y_list, y_test)

    print("训练集合上RMSE = {:.3f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("训练集合上R^2 = {:.3f}".format(r2_score(y_test, y_pred)))



if __name__ == '__main__':
    main()
