import numpy as np
from numpy import genfromtxt


def categorical_data_transform(dataPath,columnId):
    rawdata = genfromtxt(dataPath, delimiter=',')
    X = rawdata[:, columnId]
    set_X = list(set(X))
    dummy_column_header = {}
    # construct header
    for i in range(len(set_X)):
        assert not np.isnan(set_X[i]),"input contains non numerical value !"
        dummy_column_header[set_X[i]] = i
    dummy_column_matrix = np.zeros((len(X),len(set_X)))

    # fill in dummy columns
    for i in range(len(X)):
        dummy_column_matrix[i][dummy_column_header[X[i]]] = 1

    # merge output
    output = rawdata[:, :columnId]
    # insert the dummy column matrix
    output = np.insert(output, [columnId], dummy_column_matrix, axis=1)
    # append rest of the original matrix
    index = int(dummy_column_matrix.shape[1] + rawdata.shape[1])
    output = np.append(output,rawdata[:, columnId+1:-1],axis=1)
    # append Y
    output = np.append(output,rawdata[:, -1].reshape(-1,1),axis=1)
    #print(X)
    #print(dummy_column_header)
    #print(dummy_column_matrix)
    #print(output)
    np.savetxt(dataPath.split('.')[0]+"_transformed.csv", output, delimiter=',',fmt='%f')

if __name__ == '__main__':
    dataPath = r"data\multilinear_regression.csv"
    categorical_data_transform(dataPath,1)