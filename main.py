##
import pandas as pd
import numpy as np
import xgboost
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
##
comp=pd.read_csv('X_train.csv')
x_train=pd.read_csv('X_train.csv')
x_train=x_train.drop('id',axis=1)
y_train=pd.read_csv('y_train.csv',usecols=['y'])
#fills missing values with the value of another sample with closest label
def fill_nan(x_train,y_train):
    x_train['label']=y_train
    x_train=x_train.sort_values(by=['label'])
    x_train=x_train.fillna(method='ffill')
    x_train=x_train.fillna(method='bfill')
    y_train=x_train.iloc[:,-1]
    x_train=x_train.drop('label',axis=1)
    x_train=x_train.to_numpy()
    y_train=y_train.to_numpy()

    y_train=y_train.flatten()
    return x_train,y_train

x_train,y_train=fill_nan(x_train,y_train)
##
def outlier_detection(x_train):


    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(x_train,y_train)
    pred=isf.predict(x_train)

    print(isf.score_samples(x_train))
    return pred

prediction=outlier_detection(x_train)
delsample=np.where(prediction==-1)
x_train=np.delete(x_train,delsample,axis=0)
y_train=np.delete(y_train,delsample,axis=0)

#unique, counts = np.unique(prediction, return_counts=True)

#print(dict(zip(unique, counts)))

##
def feature_select_tree(x_train,x_test):
    TOP_FEATURES = 40
    forest = ExtraTreesClassifier(n_estimators=42, max_depth=10, random_state=1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_],
        axis=0
    )
    indices = np.argsort(importances)[::-1]
    indices = indices[:TOP_FEATURES]

    print('Top features:')
    for f in range(TOP_FEATURES):
        print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

    x_smol=np.zeros((x_train.shape[0],TOP_FEATURES))

    for i in range(TOP_FEATURES):
        x_smol[:,i]=x_train[:,indices[i]]
    return x_smol
#def adaboost_feature_sel(x_train,y_train):


   # return
##
rfe = RFE(estimator=XGBRegressor(),n_features_to_select=50,step=1)

rfe.fit(x_train, y_train)
vector=rfe.support_
##
x_smol=feature_select_tree(x_train,y_train)
X_train, X_test, y_train, y_test = train_test_split( x_smol, y_train, test_size=0.2, random_state=42)

lin=LinearRegression().fit(X_train,y_train)
score=r2_score(y_test,lin.predict(X_test))

pred=lin.predict(X_test)
matrix=np.zeros((2,len(y_test)))
matrix[0,:]=pred
matrix[1,:]=y_test
