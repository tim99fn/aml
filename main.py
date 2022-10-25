##
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
##
x_train=pd.read_csv('X_train.csv')
x_train=x_train.drop('id',axis=1)


x_train=x_train.fillna(x_train.mean())
y_train=pd.read_csv('y_train.csv',usecols=['y'])
x_train=x_train.to_numpy()
y_train=y_train.to_numpy()

y_train=y_train.flatten()

TOP_FEATURES = 30
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

X_train, X_test, y_train, y_test = train_test_split( x_smol, y_train, test_size=0.2, random_state=42)

lin=LinearRegression().fit(X_train,y_train)
print(lin.score(X_test, y_test))

pred=lin.predict(X_test)
matrix=np.zeros((2,len(y_test)))
matrix[0,:]=pred
matrix[1,:]=y_test
