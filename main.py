# imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# unused imports
"""
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
"""

# import functions from other files
import missing_values as sub0
import outlier_detection as sub1
import feature_selection as sub2


def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv', usecols=['y'])
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    return x_train_, y_train_, x_test_


def make_submission(prediction_):
    dt = pd.DataFrame(data=prediction_, columns=['y'])
    dt['id'] = dt.index
    dt = dt[['id', 'y']]
    dt.to_csv('submission.csv', header=True, index=False)


def standardization(x_data_):
    scaler_ = StandardScaler().fit(x_data_)
    x_data_ = scaler_.transform(x_data_)
    return x_data_


def center_data(x_data_):
    x_centered = x_data_.apply(lambda x: x - x.mean())
    return x_centered


# read in data
x_train, y_train, x_test = get_data()

# subtask 0: replace missing values
x_train, y_train = sub0.fill_nan(x_train, y_train)
x_test = x_test.fillna(x_test.median())  # for the training set use median because we don't have labels

# naive feature deletion
x_train, x_test = sub2.remove_std_zero_features(x_train, x_test)  # remove features with zero std_deviation
x_train, x_test = sub2.remove_uniform_features(x_train, x_test)  # remove features with uniform distribution

# standardization
x_train = standardization(x_train)
x_test = standardization(x_test)

# subtask 1: outlier detection
x_train, y_train = sub1.outlier_detection_gmm(x_train, y_train, 200, plot=False)  # pca to 200 explains 75% of variance

# subtask 2: feature selection
print("Feature Selection:")
# print("features before pca: ", x_train.shape[1])
# x_train, x_test = sub2.pca_reduction(x_train, x_test, 400)
# print("features after pca: ", x_train.shape[1])
x_train, x_test = sub2.feature_select_tree(x_train, y_train, x_test, 110)
print("features after tree select: ", x_train.shape[1])

# train test split
x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

# fit the model
las = LassoCV(cv=10).fit(x_train, y_train)
prediction = las.predict(x_test_val)
print(r2_score(y_test_val, prediction))

# make a submission
# make_submission(prediction)
