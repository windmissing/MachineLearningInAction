# -*- coding: utf-8 -*-
import numpy as np

verbose = 0

def loadSimpData():
    datMat = np.mat([[ 1. , 2.1],
    [ 2. , 1.1],
    [ 1.3, 1. ],
    [ 1. , 1. ],
    [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def calculateAccuray(predict, target):
    s = 0
    for i in range(predict.shape[0]):
        if predict[i] == target[i]:
            s += 1
    return s/predict.shape[0]

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        dataList = [1] + [float(data) for data in line.strip().split('\t')]  # 应该增加一个x0 =1，但书上没有[1]
        dataMat.append(dataList[0:-1])
        labelMat.append(dataList[-1])
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    total = np.array([np.sum(np.array(classLabels) == 1), np.sum(np.array(classLabels) == -1)])
    cur = total.copy()
    rocNode = [cur/total]
    sortedIndices = predStrengths.argsort()
    for index in sortedIndices:
        if classLabels[index] == 1:
            cur[0] -=1
        else:
            cur[1] -=1
        rocNode.append(cur/total)
    if verbose:
        print (np.array(rocNode))
        print (predStrengths)
        print (sortedIndices)
    plt.plot(np.array(rocNode)[:,1], np.array(rocNode)[:, 0])
    plt.show()