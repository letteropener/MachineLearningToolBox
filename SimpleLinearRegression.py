import numpy as np

# y = b0 + b1*X
#Mean Square Error
# min Sum(y_i - y_hat)^2
# b1 = sum[(x_i - x_mean)*(y_i - y_mean)] / sum[(x_i - x_mean)^2]
# b0 = y_mean - b1 * x_mean


def fitSLR(x,y):
    n = len(x)
    dinominator = 0
    numerator = 0
    for i in range(0,n):
        numerator += (x[i]- np.mean(x))* (y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x)) ** 2

    print("numerator: ", numerator)
    print("dinominator: ",dinominator)

    b1 = numerator / float(dinominator)
    b0 = np.mean(y) - b1 * np.mean(x)
    return b0, b1

def predict(x,b0,b1):
    return b0 + x*b1

x = [1,3,2,1,3]
y = [14,24,18,17,27]
b0, b1 = fitSLR(x,y)
print("intecept: ",b0, " slope: ",b1)

x_test = 6
y_test = predict(x_test,b0,b1)
print("y_test: ",y_test)