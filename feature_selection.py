import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import chisquare
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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


def remove_std_zero_features(x_train_, x_test_):
    """ removes features which have zero std_deviation
     :parameter:
     x_train_ : data frame of training set
     x_test_ : data frame of test set

     :return:
     x_train_ : training set with the std_deviation == 0 features removed
     x_test_ : test set with the same features as for training set removed

     Notes:
     ---------
     This function removes all the features from the training set that have std_deviation == 0.
     It also removes the corresponding features in the training set
     """
    zero_std = (x_train_.std() > 0.0)
    print("we remove ", x_train_.shape[1] - zero_std.sum(), "features which have std_deviation == 0")
    x_train_ = x_train_.loc[:, zero_std]
    x_test_ = x_test_.loc[:, zero_std]
    return x_train_, x_test_


def remove_uniform_features(x_train_, x_test_):
    """ removes features which are uniform distributed
    :parameter:
    x_train_ : data frame of training set
    x_test_ : data frame of test set

    :return:
    x_train_ : training set with the uniform features removed
    x_test_ : test set with the same features as for training set removed

    Notes:
    ---------
    This function removes all uniform like distributed features from the training set
    and the corresponding features from the test set as well.
    This is done using a chi_squared test and excludes all features with p_value > 0.05
    """
    x = x_train_.to_numpy()
    chi_test = np.zeros(x.shape[1])  # stores a 1 if the feature is uniform like

    # does the xhi_squared test for each feature and stores a 1 in test if p_value > 0.05
    for i in range(x.shape[1]):
        hist = np.histogram(x[:, i], bins=20)[0]
        hist = np.delete(hist, hist.argmax())
        unif = np.ones_like(hist) * hist.sum() / hist.shape
        if chisquare(hist, unif).pvalue > 0.05:
            chi_test[i] = 1

    print("we remove ", chi_test.sum(), "uniform looking features")
    unif_cols_indicies = chi_test.nonzero()[0]  # indices of uniform features
    unif_cols_names = x_train_.columns[unif_cols_indicies]  # data frame column names of uniform features
    x_train_ = x_train_.drop(unif_cols_names, axis=1)  # drops these columns
    x_test_ = x_test_.drop(unif_cols_names, axis=1)

    return x_train_, x_test_

def feature_select_bic(x_train_, y_train_, x_test_):

    model = sm.OLS(y_train_,x_train_).fit()
    bic = model.bic
    return bic
