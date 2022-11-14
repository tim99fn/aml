# imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic
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
x_train, y_train = sub1.novelty_svm(x_train, y_train, x_test, 5)
x_train, y_train = sub1.age_outlier(x_train, y_train, 12)

# standardization
x_train = standardization(x_train)
x_test = standardization(x_test)

# subtask 2: feature selection
x_train, x_test = sub2.lasso_lars(x_train, y_train, x_test, 'bic')

# Model evaluation
# x_train, x_test_val, y_train, y_test_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
gpr = GaussianProcessRegressor(kernel=RBF()+Matern(), random_state=42, normalize_y=True).fit(x_train, y_train)
prediction, pred_std = gpr.predict(x_test, return_std=True)
make_submission(prediction)
# score = r2_score(y_test_val, prediction)
# print(score)
