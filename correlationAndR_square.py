import numpy as np
import math

def computeCorrelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0,len(X)):
        diffXBar = X[i] - xBar
        diffYBar = Y[i] - yBar
        SSR += diffXBar * diffYBar
        varX += (diffXBar)**2
        varY += (diffYBar)**2

    SST = math.sqrt(varX*varY)
    return SSR / SST

# polynomial regression
# b1*x1 + b2*x2 + .. bn*xn for degree = 1
# x ^ 2 for degree = 2
def polyfit(x,y,degree):
    results = {}
    coeffs = np.polyfit(x,y,degree)
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values and mean
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    results['determination'] = ssreg/sstot
    return results

testX = [1,3,8,7,9]
testY = [10,12,24,21,34]

print("r:",computeCorrelation(testX,testY))
print("r^2:",computeCorrelation(testX,testY)**2)
print("polyfit:",polyfit(testX,testY,1)['determination'])