# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import operator

# inX：the data to be predicted
# dataSet：训练集
# labels：训练集对应的标签
#datasetSize = dataSet.shape[0]   # 二维数组，行数=array.shape[0]，列数=array.shape[1]
def classify0(inX, dataSet, labels, k):
    ret = []
    for x in inX:
        ret.append(classifyForOneData(x, dataSet, labels, k))
    return np.array(ret)

# For every point in our dataset:
#     calculate the distance between inX and the current point
#     sort the distances in increasing order
#     take k items with lowest distances to inX
#     find the majority class among these items
#     return the majority class as our prediction for the class of inX
def classifyForOneData(x, dataSet, labels, k):
    diff = dataSet - x
    distance = np.sum(diff**2, axis=1) **0.5
    distanceIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        label = labels[distanceIndex[i]]
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest(dataSet, labels):
    error = 0
    total = 0
    testFileList = listdir("testDigits")
    for file in testFileList:
        actualLabel = int(file.split('_')[0])
        expectLabel = classify0(img2vector('testDigits/'+file), dataSet, labels, 3)
        print ("actualLabel = " + str(actualLabel) + ", expectLabel = " + str(expectLabel))
        total += 1
        if actualLabel != expectLabel: error+=1
    return error / total
