from util import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def load_numeric_variables():
    processed_df = joinFilesOnColumn('data/USD_CAD.csv', 'data/Oil_Future.csv', 'data/SP500_Future.csv',
                                     'data/Gold_Future.csv'
                                     , 'Date')
    X,Y = returnDFinX_Y(processed_df)
    numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
    #print(X[numeric_variables].head())
    return [X[numeric_variables],Y]


def benchmark(X_train,Y_train,X_test,Y_test):
    model = LinearRegression()
    model.fit(X_train,Y_train)
    Y_Pred = model.predict(X_test)
    score = r2_score(Y_test,Y_Pred)
    print(score)
    correct = 0
    for x in range(len(Y_test)):
        if np.sign(Y_Pred[x]) == np.sign(Y_test[x]):
            correct += 1
    print(str(correct))

def RandomForestModel(X,Y):
    model = RandomForestRegressor(n_estimators=100,oob_score=True,random_state=10)
    model.fit(X,Y)
    y_oob = model.oob_prediction_
    print(y_oob)
    print("c-stat: ",roc_auc_score(Y,y_oob))


def main():
    X,Y = load_numeric_variables()
    X_train = X[X.index >= 500]
    Y_train = Y[Y.index >= 500]
    X_test = X[X.index < 500]
    Y_test = Y[Y.index < 500]
    benchmark(X_train,Y_train,X_test,Y_test)

main()