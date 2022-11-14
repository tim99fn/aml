# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic
from sklearn.model_selection import LeaveOneOut, KFold


# import functions from other files
import missing_values as sub0
import outlier_detection as sub1
import feature_selection as sub2


def get_data():
    x_train_ = pd.read_csv('X_train.csv').drop('id', axis=1)
    y_train_ = pd.read_csv('y_train.csv').drop('id', axis=1)
    x_test_ = pd.read_csv('X_test.csv').drop('id', axis=1)
    return x_train_, y_train_, x_test_


def make_submission(prediction_, name='submission.csv'):
    dt = pd.DataFrame(data=prediction_, columns=['y'])
    dt['id'] = dt.index
    dt = dt[['id', 'y']]
    dt.to_csv(name, header=True, index=False)


def standardization(x):
    scaler_ = StandardScaler()
    return scaler_.fit_transform(x)


def robust_transform(x):
    transformer = RobustScaler()
    return transformer.fit_transform(x)


# read in data
x_train, y_train, x_test = get_data()

# subtask 0: replace missing values
"""
x_train, y_train = sub0.fill_nan(x_train, y_train)
x_test = x_test.fillna(x_test.median())  # for the training set use median because we don't have labels
"""
x_train['label'] = y_train
x_train = sub0.knn_imputer(x_train, metr=sub0.age_similarity)
x_train = np.delete(x_train, -1, 1)
x_test = sub0.knn_imputer(x_test)
y_train = y_train.to_numpy().reshape(-1)


# naive feature deletion
x_train, x_test = sub2.remove_std_zero_features(x_train, x_test)  # remove features with zero std_deviation
x_train, x_test = sub2.remove_uniform_features(x_train, x_test)  # remove features with uniform distribution

# standardization
x_train = standardization(x_train)
x_test = standardization(x_test)

# subtask 1: outlier detection
# x_train, y_train = sub1.outlier_detection_gmm(x_train, x_test, y_train, 400, 5, plot=False)
x_train, y_train = sub1.novelty_svm(x_train, y_train, x_test, 5)
x_train, y_train = sub1.age_outlier(x_train, y_train, 12)

# standardization
x_train = standardization(x_train)
x_test = standardization(x_test)

# subtask 2: feature selection
x_train, x_test = sub2.lasso_lars(x_train, y_train, x_test, 'bic')

# Model evaluation
x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
gpr = GaussianProcessRegressor(kernel=Matern()+RBF(), random_state=42, normalize_y=True, ).fit(x_train, y_train)
prediction = gpr.predict(x_test_val)


score = r2_score(y_test_val, prediction)
print(score)
matrix = np.stack((prediction, y_test_val))
diff = np.abs(prediction-y_test_val)


plt.figure()
plt.title("histogram of labels vs predictions")
plt.hist(prediction, bins=55, alpha=0.6, label='prediction')
plt.hist(y_test_val, bins=55, alpha=0.6, label='true labels')
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.title("error vs age")
plt.scatter(y_test_val, diff, label='error')
plt.legend(loc='upper right')
plt.show()

# make a submission
# make_submission(prediction)
print("that's it")


"""
diff = np.zeros_like(y_train)
X = x_train
y = y_train
# train model the first time

# plot outliers

# loo = LeaveOneOut()
loo = KFold(n_splits=100)
loo.get_n_splits(X)
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    gpr = GaussianProcessRegressor(kernel=Matern() + RBF(), random_state=42, normalize_y=True, ).fit(X_train, Y_train)
    prediction = gpr.predict(X_test)
    diff[test_index] = np.abs(prediction-Y_test)

print((diff > 10).sum())

make_submission(diff, 'outlier_score.csv')
"""