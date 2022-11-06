import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


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
