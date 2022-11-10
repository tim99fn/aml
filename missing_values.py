from sklearn.impute import KNNImputer
import pandas as pd


# fills missing values with the value of another sample with the closest label
def fill_nan(x_train_, y_train_):
    x_train_['label'] = y_train_
    x_train_ = x_train_.sort_values(by=['label'])
    x_train_ = x_train_.fillna(method='ffill')
    x_train_ = x_train_.fillna(method='bfill')
    y_train_ = x_train_['label']
    x_train_ = x_train_.drop('label', axis=1)
    y_train_ = y_train_.to_numpy()
    return x_train_, y_train_


def knn_imputer(x_train_):
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    return imputer.fit_transform(x_train_)