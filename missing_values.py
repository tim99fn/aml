from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


# fills missing values with the value of another sample with the closest label
def fill_nan(x_train_, y_train_):
    x_train_['label'] = y_train_
    x_train_ = x_train_.sort_values(by=['label'])
    x_train_ = x_train_.fillna(method='ffill')
    x_train_ = x_train_.fillna(method='bfill')
    y_train_ = x_train_['label']
    x_train_ = x_train_.drop('label', axis=1)
    y_train_ = y_train_.to_numpy().reshape(-1)
    return x_train_, y_train_


def knn_imputer(x_train_, metr='nan_euclidean'):
    imputer = KNNImputer(n_neighbors=5, weights='distance', metric=metr)
    return imputer.fit_transform(x_train_)


def age_similarity(x, y, missing_values=np.nan):
    return (x[-1] - y[-1])**2
