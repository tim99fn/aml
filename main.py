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

# read in data
x_train, y_train, x_test = get_data()

# subtask 0: replace missing values
x_train, y_train = sub0.fill_nan(x_train, y_train)
x_test = x_test.fillna(x_test.median())  # for the training set use median because we don't have labels

# naive feature deletion
x_train, x_test = sub2.remove_std_zero_features(x_train, x_test)  # remove features with zero std_deviation
x_train, x_test = sub2.remove_uniform_features(x_train, x_test)  # remove features with uniform distribution

# normalization
x_train = normalization(x_train)
x_test = normalization(x_test)
y_normed = (y_train-np.mean(y_train))/np.std(y_train)

# subtask 1: outlier detection
x_train, y_train = sub1.outlier_detection_gmm(x_train, y_train, y_normed, 50, plot=True)
#x_train, y_train = sub1.outlier_detection(x_train, y_train)

# again normalization .... is this needed?
"""
x_train = normalization(x_train)
x_test = normalization(x_test)
"""

# subtask 2: feature selection
x_train, x_test = sub2.feature_select_tree(x_train, y_train, x_test, 500)

#
x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

# fit the model
las = LassoCV(cv=10).fit(x_train, y_train)
prediction = las.predict(x_test_val)
print(r2_score(y_test_val,prediction))

# make a submission
# make_submission(prediction)
