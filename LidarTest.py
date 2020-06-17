import numpy as np
import pprint
from math import sqrt

def editDistance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])
    return distance

def euclideanDistance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def getNeighborsEditDistance(trainingSet, testRow, numNeighbors):
    distances = list()
    for trainingRow in trainingSet:
        if not (trainingRow == testRow):
            dist = editDistance(testRow, trainingRow)
            distances.append((trainingRow, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(numNeighbors):
        neighbors.append(distances[i][1])
    return neighbors

def getNeighborsEuclideanDistance(trainingSet, testRow, numNeighbors):
    distances = list()
    for trainingRow in trainingSet:
        if not (trainingRow == testRow):
            dist = euclideanDistance(testRow, trainingRow)
            distances.append((trainingRow, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(numNeighbors):
        neighbors.append(distances[i][1])
    return neighbors

def getClassificationByAverage(positiveNeighbors, negativeNeighbors):
    if (average(positiveNeighbors) < average(negativeNeighbors)):
        return 1
    elif (average(negativeNeighbors) < average(positiveNeighbors)):
        return 0
    else:
        return -1

def getClassificationByNearest(positiveNeighbors, negativeNeighbors):
    if (positiveNeighbors[0] < negativeNeighbors[0]):
        return 1
    elif (negativeNeighbors[0] < positiveNeighbors[0]):
        return 0
    else:
        return -1
    


def average(lst):
    return sum(lst)/len(lst)


def generateConfusionMatrix(positiveDataSet, negativeDataSet, getNeighbors, getClassification):
    target_numPredictedAsTarget = 0
    target_numPredictedAsNotTarget = 0
    notTarget_numPredictedAsTarget = 0
    notTarget_numPredictedAsNotTarget = 0
    numNeighbors = 9 #use all values in comparison
    
    #cycle through positive data set and change values accordingly
    for i in range(len(positiveDataSet)):
        positiveNeighbors = getNeighbors(positiveDataSet, positiveDataSet[i], numNeighbors)
        negativeNeighbors = getNeighbors(negativeDataSet, positiveDataSet[i], numNeighbors)
        classification = getClassification(positiveNeighbors, negativeNeighbors)
        if not classification == -1:
            if classification == 1:
                target_numPredictedAsTarget += 1
            else:
                target_numPredictedAsNotTarget +=1
    
    #cycle through negative data set and change values accordingly
    for i in range(len(negativeDataSet)):
        positiveNeighbors = getNeighbors(negativeDataSet, negativeDataSet[i], numNeighbors)
        negativeNeighbors = getNeighbors(positiveDataSet, negativeDataSet[i], numNeighbors)
        classification = getClassification(positiveNeighbors, negativeNeighbors)
        if not classification == -1:
            if classification == 1:
                notTarget_numPredictedAsNotTarget += 1
            else:
                notTarget_numPredictedAsTarget +=1
                
    return [[target_numPredictedAsTarget, target_numPredictedAsNotTarget],[notTarget_numPredictedAsTarget, notTarget_numPredictedAsNotTarget]]
                
    

actualPeaks_turb3 = np.loadtxt("turb_3_actual_peaks.csv", delimiter=",", skiprows=1).tolist()
falsePeaks_turb3 = np.loadtxt("turb_3_false_peaks.csv", delimiter=",", skiprows=1).tolist()
actualPeaks_turb6 = np.loadtxt("turb_6_actual_peaks.csv", delimiter=",", skiprows=1).tolist()
falsePeaks_turb6 = np.loadtxt("turb_6_false_peaks.csv", delimiter=",", skiprows=1).tolist()

print("Confusion Matrix for Turb 3 running EditDistance with Average on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb3, falsePeaks_turb3, getNeighborsEditDistance, getClassificationByAverage))
print("Confusion Matrix for Turb 6 running EditDistance with Average on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb6, falsePeaks_turb6, getNeighborsEditDistance, getClassificationByAverage))
print("\n")
print("Confusion Matrix for Turb 3 running EuclideanDistance with Average on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb3, falsePeaks_turb3, getNeighborsEuclideanDistance, getClassificationByAverage))
print("Confusion Matrix for Turb 6 running EuclideanDistance with Average on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb6, falsePeaks_turb6, getNeighborsEuclideanDistance, getClassificationByAverage))

print("\n\n\n")

print("Confusion Matrix for Turb 3 running EditDistance on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb3, falsePeaks_turb3, getNeighborsEditDistance, getClassificationByAverage))
print("Confusion Matrix for Turb 6 running EditDistance on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb6, falsePeaks_turb6, getNeighborsEditDistance, getClassificationByAverage))
print("\n")
print("Confusion Matrix for Turb 3 running EuclideanDistance on Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb3, falsePeaks_turb3, getNeighborsEuclideanDistance, getClassificationByNearest))
print("Confusion Matrix for Turb 6 running EuclideanDistance with Set:")
pprint.pprint(generateConfusionMatrix(actualPeaks_turb6, falsePeaks_turb6, getNeighborsEuclideanDistance, getClassificationByNearest))
    