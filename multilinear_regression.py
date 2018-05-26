from numpy import  genfromtxt
import numpy as np
from  sklearn import datasets, linear_model
from categorical_data_2_oneHot import categorical_data_transform

dataPath = r"data\multilinear_regression.csv"
categorical_data_transform(dataPath,columnId=2)
transfomed_dataPath = r"data\multilinear_regression_transformed.csv"
deliveryData = genfromtxt(transfomed_dataPath,delimiter=',')
print("transformed data")
print(deliveryData)

X = deliveryData[:,:-1]
Y = deliveryData[:,-1]

regr = linear_model.LinearRegression()
regr.fit(X,Y)

print("coefficients: ")
print(regr.coef_)
print("intercept: ")
print(regr.intercept_)

# make a prediction
xPred = np.array([102,6,1,0,0]).reshape(1,-1)
yPred = regr.predict(xPred)
print("predicted y: ")
print(yPred)
