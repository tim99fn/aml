# imports
import pandas as pd
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

# import functions from other files
import missing_values as sub0
import outlier_detection as sub1
import feature_selection as sub2


def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv', usecols=['y'])
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    x_test_ = x_test_.fillna(x_test_.mean())
    x_test_ = x_test_.to_numpy()
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
x_train, y_train, test = get_data()

# subtask 0: replace missing values
x_train, Y = sub0.fill_nan(x_train, y_train)

# normalization
x_train = standardization(x_train)
test = standardization(test)

# subtask 1: outlier detection
x_train, Y = sub1.outlier_detection_gmm(x_train, Y,50, plot=True)

# again normalization
x_train = standardization(x_train)
test = standardization(test)

# subtask 3: feature selection
x_smol, new_test = sub2.feature_select_tree(x_train, Y, test, 500)

##
# X_train, X_test, y_train, y_test = train_test_split( x_smol, Y, test_size=0.15, random_state=42)

# fit the model
las = LassoCV(cv=10).fit(x_smol, Y)
prediction = las.predict(new_test)

# make a submission
make_submission(prediction)
