import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))

class NN:
    def __init__(self,layers,activation='tanh'):
        """
        :param layers: A list containing the number of neuron units in each layer
        Should be at least two values
        :param activation: The activation function to be used.
        Can e "Logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1,len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)

    def fit(self,X,y,learning_rate=0.02,epochs=100000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0],X.shape[1]+1]) # add bias unit
        temp[:,0:-1] = X
        X =temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            error = y[i] - a[-1]  # diff between label and pred
            # delta at output layer
            deltas = [error*self.activation_deriv(a[-1])]

            # start backpropagation
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta =np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        x = temp
        #print(x)
        a = [x]
        for l in range(0,len(self.weights)):
            a.append(self.activation(np.dot(a[l],self.weights[l])))
        return a[-1]

