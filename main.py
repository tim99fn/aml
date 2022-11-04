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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV


##
def get_data():
    comp=pd.read_csv('X_train.csv')
    x_train=pd.read_csv('X_train.csv')
    x_train=x_train.drop('id',axis=1)
    y_train=pd.read_csv('y_train.csv',usecols=['y'])
    test=pd.read_csv('X_test.csv')
    test=test.drop('id',axis=1)
    test=test.fillna(test.mean())
    test=test.to_numpy()
    return x_train,y_train,test

##
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


#test,y_train=fill_nan(test,y_train)
#x_train= normalize(x_train,norm='l1',axis=0)
def make_submission(prediction):
    dt = pd.DataFrame(data=prediction,columns=['y'])
    dt['id']=dt.index
    dt=dt[['id','y']]
    dt.to_csv('submission.csv',header=True,index=False)

def outlier_detection(x_train,Y):


    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(x_train,Y)
    pred=isf.predict(x_train)

    #print(isf.score_samples(x_train))
    delsample=np.where(pred==-1)
    print(delsample)
    x_train=np.delete(x_train,delsample,axis=0)
    Y=np.delete(Y,delsample,axis=0)
    return x_train,Y




#unique, counts = np.unique(prediction, return_counts=True)

#print(dict(zip(unique, counts)))


def feature_select_tree(x_train,y_train,test,TOP_FEATURES):

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
    new_test=np.zeros((test.shape[0],TOP_FEATURES))

    for i in range(TOP_FEATURES):
        x_smol[:,i]=x_train[:,indices[i]]
        new_test[:,i]=test[:,indices[i]]
    return x_smol,new_test
#def adaboost_feature_sel(x_train,y_train):


   # return

##
x_train,y_train,test=get_data()
x_train,Y=fill_nan(x_train,y_train)
scaler=StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
test=scaler.transform(test)
x_train,Y=outlier_detection(x_train,Y)
scaler=StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
test=scaler.transform(test)
x_smol,new_test=feature_select_tree(x_train,Y,test,500)

##
#x_norm=normalize(x_smol,norm='l1',axis=0)
##
#X_train, X_test, y_train, y_test = train_test_split( x_smol, Y, test_size=0.15, random_state=42)
las= LassoCV(cv=10).fit(x_smol,Y)

prediction=las.predict(new_test)

##
make_submission(prediction)
