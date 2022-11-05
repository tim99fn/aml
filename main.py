# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# unused imports
"""
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
"""


##


def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv', usecols=['y'])
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    x_test_ = x_test_.fillna(x_test_.mean())
    x_test_ = x_test_.to_numpy()
    return x_train_, y_train_, x_test_


# fills missing values with the value of another sample with the closest label


def fill_nan(x_train_, y_train_):
    x_train_['label'] = y_train_
    x_train_ = x_train_.sort_values(by=['label'])
    x_train_ = x_train_.fillna(method='ffill')
    x_train_ = x_train_.fillna(method='bfill')
    y_train_ = x_train_.iloc[:, -1]
    x_train_ = x_train_.drop('label', axis=1)
    x_train_ = x_train_.to_numpy()
    y_train_ = y_train_.to_numpy()

    y_train_ = y_train_.flatten()
    return x_train_, y_train_


# test,y_train=fill_nan(test,y_train)
# x_train= normalize(x_train,norm='l1',axis=0)


def make_submission(prediction_):
    dt = pd.DataFrame(data=prediction_, columns=['y'])
    dt['id'] = dt.index
    dt = dt[['id', 'y']]
    dt.to_csv('submission.csv', header=True, index=False)


##


def outlier_detection(x_train_, y_):
    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(x_train_, y_)
    pred = isf.predict(x_train_)
    # print(isf.score_samples(x_train))
    delsample = np.where(pred == -1)
    print(delsample)
    x_train_ = np.delete(x_train_, delsample, axis=0)
    y_ = np.delete(y_, delsample, axis=0)
    return x_train_, y_


# unique, counts = np.unique(prediction, return_counts=True)

# print(dict(zip(unique, counts)))


def feature_select_tree(x_train_, y_train_, test_, top_features_):
    forest = ExtraTreesClassifier(n_estimators=42, max_depth=10, random_state=1)
    forest.fit(x_train_, y_train_)
    importances = forest.feature_importances_
    np.std(
        [tree.feature_importances_ for tree in forest.estimators_],
        axis=0
    )
    indices = np.argsort(importances)[::-1]

    indices = indices[:top_features_]

    print('Top features:')
    for f in range(top_features_):
        print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

    x_smol_ = np.zeros((x_train_.shape[0], top_features_))
    new_test_ = np.zeros((test_.shape[0], top_features_))

    for i in range(top_features_):
        x_smol_[:, i] = x_train_[:, indices[i]]
        new_test_[:, i] = test_[:, indices[i]]
    return x_smol_, new_test_


# def adaboost_feature_sel(x_train,y_train):


# return

##

# read in data
x_train, y_train, test = get_data()

# subtask 0: replace missing values
x_train, Y = fill_nan(x_train, y_train)

# normalization
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
test = scaler.transform(test)

# subtask 1: outlier detection
x_train, Y = outlier_detection(x_train, Y)

# again normalization
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
test = scaler.transform(test)

# subtask 3: feature selection
x_smol, new_test = feature_select_tree(x_train, Y, test, 500)
##
# x_norm=normalize(x_smol,norm='l1',axis=0)
##
# X_train, X_test, y_train, y_test = train_test_split( x_smol, Y, test_size=0.15, random_state=42)

# fit the model
las = LassoCV(cv=10).fit(x_smol, Y)
prediction = las.predict(new_test)
##
make_submission(prediction)
