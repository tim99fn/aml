import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import chisquare
from sklearn.linear_model import LassoLarsIC
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold


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

    """
    print('Top features:')
    for f in range(top_features_):
        print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))
    """

    x_smol_ = np.zeros((x_train_.shape[0], top_features_))
    new_test_ = np.zeros((test_.shape[0], top_features_))

    for i in range(top_features_):
        x_smol_[:, i] = x_train_[:, indices[i]]
        new_test_[:, i] = test_[:, indices[i]]
    return x_smol_, new_test_


def remove_std_zero_features(x_train_, x_test_):
    selector = VarianceThreshold()
    return selector.fit_transform(x_train_), selector.transform(x_test_)


def remove_uniform_features(x, x_test_):

    chi_test = np.zeros(x.shape[1])  # stores a 1 if the feature is uniform like
    # does the xhi_squared test for each feature and stores a 1 in test if p_value > 0.05
    for i in range(x.shape[1]):
        hist = np.histogram(x[:, i], bins=20)[0]
        hist = np.delete(hist, hist.argmax())
        unif = np.ones_like(hist) * hist.sum() / hist.shape
        if chisquare(hist, unif).pvalue > 0.05:
            chi_test[i] = 1

    print("we remove ", chi_test.sum(), "uniform looking features")
    elim = chi_test.nonzero()[0]  # indices of uniform features
    x = np.delete(x, elim, axis=1)
    x_test_ = np.delete(x_test_, elim, axis=1)
    return x, x_test_


def feature_select_bic(x_train_, y_train_, x_test_):

    # initialize convergence parameter
    epsilon = 0.0000001

    # initialize bic value
    bic_min1 = pow(2,64)
    bic_min2 = 0

    # initialize best feature array
    best_features = []
    best_feature = 0

    # iterate over all possible feature combination by tree method until convergence
    i=0
    while abs(bic_min2-bic_min1)>epsilon and i<800:
        i+=1
        bic_min2 = bic_min1
        for j in range(x_train_.shape[1]):
            if j not in best_features:
                model = sm.OLS(y_train_, sm.add_constant(x_train_[:, best_features + [j]])).fit()
                bic = model.bic
                if bic < bic_min1:
                    bic_min1 = bic
                    best_feature = j
        best_features = best_features + [best_feature]
    return x_train_[:,best_features],x_test_[:,best_features]


def Lasso_feature_extraction(x_train, x_test, y_train):
    # fit the model
    las = LassoCV(cv=10).fit(x_train, y_train)
    coef = np.where(las.coef_ != 0)
    x_train = x_train[:, coef]
    x_test = x_test[:, coef]
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    return x_train,x_test


def pca_reduction(x_train_, x_test_, dimensions):
    # extract Principle Components
    pca = PCA(n_components=dimensions, svd_solver='full')
    # pca.fit already returns data projected on lower dimensions
    return pca.fit_transform(x_train_), pca.fit_transform(x_test_)


def lasso_lars(x_train_, y_train_, x_test_, crit='bit'):
    reg = LassoLarsIC(criterion=crit, normalize=False, max_iter=1000).fit(x_train_, y_train_)
    cof = reg.coef_.nonzero()[0]
    return x_train_[:, cof], x_test_[:, cof]

