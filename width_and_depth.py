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

def main():
    data_train = pd.read_csv('./nin.cifar10_svhn.csv')
    data_train = data_train[np.logical_and(data_train['hp.dataset'] == 'svhn', data_train['is.converged'] == True,
                                           data_train['is.high_train_accuracy'] == True)]
    data_x = data_train.loc[:, ['hp.model_width']]
    data_y = data_train.loc[:, ['gen.gap']]
    tau_gen_gap, p_value = stats.kendalltau(data_y, data_x)
    print(tau_gen_gap)





if __name__ == '__main__':
    main()
