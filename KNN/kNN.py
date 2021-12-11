# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import operator

# inX：the data to be predicted
# dataSet：训练集
# labels：训练集对应的标签
#datasetSize = dataSet.shape[0]   # 二维数组，行数=array.shape[0]，列数=array.shape[1]
def classify(train_X, train_y, test_X, k):
    ret = []
    for x in test_X:
        ret.append(classifyForOne(train_X, train_y, x, k))
    return np.array(ret)

# For every point in our dataset:
#     calculate the distance between inX and the current point
#     sort the distances in increasing order
#     take k items with lowest distances to inX
#     find the majority class among these items
#     return the majority class as our prediction for the class of inX
from collections import Counter
def classifyForOne(train_X, train_y, x, k):
    diff = train_X - x
    distance = np.sum(diff**2, axis=1)**0.5
    index = np.argsort(distance, axis=0)
    topK_y = train_y[index[:k]]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]

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
