import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from Setting import *

def main():
    data_train = pd.read_csv('./nin.cifar10_svhn.csv')
    data_train = data_train[np.logical_and(data_train['hp.dataset']==task, data_train['is.converged']==True, data_train['is.high_train_accuracy']==True)]
    df = data_train[csv_list]
    df.to_csv('./spss_cifar10.csv', index=None)


if __name__ == '__main__':
    main()
