import math
import operator
from util import splitDataset,joinFilesOnColumn


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet,testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct +=1
    return (correct/float(len(testSet))) * 100.0


def main():

    processed_df = joinFilesOnColumn('data/USD_CAD.csv','data/Oil_Future.csv','data/SP500_Future.csv','data/Gold_Future.csv'
    ,'Date')
    #print(processed_df)
    for epoch in range(1):
        trainingSet = []
        testSet = []
        split = 0.85
        splitDataset(processed_df,split,trainingSet,testSet)
        print('Train set: '+ repr(len(trainingSet)))
        print('Test set: '+ repr(len(testSet)))
        # generate predictions
        predictions = []
        k = 10
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet,testSet[x],k)
            result = getResponse(neighbors)
            predictions.append(result)
            #print('> predicted='+repr(result)+', actual='+repr(testSet[x][-1]))
        accuracy = getAccuracy(testSet,predictions)
        print('epoch '+ str(epoch)+' Accuracy: ' + repr(accuracy) + '%')


main()