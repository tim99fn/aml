# imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
# unused imports
##
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
    y_train_ = pd.read_csv('y_train.csv').drop('id', axis=1)
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    return x_train_, y_train_, x_test_


def make_submission(prediction_):
    dt = pd.DataFrame(data=prediction_, columns=['y'])
    dt['id'] = dt.index
    dt = dt[['id', 'y']]
    dt.to_csv('submission.csv', header=True, index=False)


def standardization(x):
    scaler_ = StandardScaler()
    return scaler_.fit_transform(x)


def robust_transform(x):
    transformer = RobustScaler()
    return transformer.fit_transform(x)


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
x_train, y_train = sub1.outlier_detection_gmm(x_train, x_test, y_train, 400, 10, plot=False)

# subtask 2: feature selection
x_train, x_test = sub2.lasso_lars(x_train, y_train, x_test)


# Model evaluation
x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
gpr = GaussianProcessRegressor(kernel=Matern()+RBF(), random_state=42,normalize_y=False).fit(x_train, y_train)
prediction = gpr.predict(x_test_val)
score = r2_score(y_test_val, prediction)
print(score)

matrix = np.stack((prediction,y_test_val))
# make a submission
#make_submission(prediction)
