import numpy as np
from sklearn.ensemble import IsolationForest


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
