# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
import operator
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def createDataSet():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.2, random_state=666)

def plotData(train_X, train_y, test_X):
    plt.scatter(train_X[train_y==0, 0], train_X[train_y==0, 1], color = 'g')
    plt.scatter(train_X[train_y==1, 0], train_X[train_y==1, 1], color = 'r')
    plt.scatter(test_X[:,0], test_X[:,1], color = 'b')
    plt.show()


def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)

    returnMat = np.zeros((numberOfLines, 3))  # 初始化样本特征矩阵
    classLabelVector = []  # 初始化化输出标记列表
    index = 0

    for line in lines:  # 依次处理每一行
        line = line.strip()  # 去掉行尾的换行符
        listFromLine = line.split('\t')  # 提取每个特征
        returnMat[index, :] = listFromLine[0:-1]  # 提取特征向量，并加入到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 提取输出标记，并加入输出列表中

        index += 1
    return np.array(returnMat), np.array(classLabelVector)

def plotDatingData(train_X, train_y):
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(131)
    ax.scatter(train_X[:,0], train_X[:,1],15.0*np.array(train_y), 15.0*np.array(train_y))
    ax = fig.add_subplot(132)
    ax.scatter(train_X[:,1], train_X[:,2],15.0*np.array(train_y), 15.0*np.array(train_y))
    ax = fig.add_subplot(133)
    ax.scatter(train_X[:,0], train_X[:,2],15.0*np.array(train_y), 15.0*np.array(train_y))
    plt.show()


def autoNorm(dataSet):
    max = np.max(dataSet, axis=0)
    min = np.min(dataSet, axis=0)
    ranges = max-min
    return ((dataSet -min) / ranges), ranges, min

def splitTrainAndTest(X, y, testRatio):
    numTestCase = int(X.shape[0] * testRatio)
    return X[numTestCase:], y[numTestCase:], X[:numTestCase], y[:numTestCase]

def calculateAccuray(predict, target):
    s = 0
    for i in range(predict.shape[0]):
        if predict[i] == target[i]:
            s += 1
    return s/predict.shape[0]


def preprocessForTestData(ffMiles, percentTats, iceCream, ranges, minVals):
    x = np.array([ffMiles, percentTats, iceCream])
    x = (x - minVals)/ranges
    inX = np.array([x])
    return inX


def labelId2Text(labelId):
    resultList = ['not at all','in small doses', 'in large doses']
    return resultList[labelId - 1]


def img2vector(filename):
    ret = []
    with open(filename) as fr:
        for line in fr.readlines():
            line = list(map(int, line[:32]))
            ret = ret + line
            # ret = list(ret) + list(line)
    return np.array(ret) #.reshape(1,-1)

from os import listdir
def downloadHandWritingData(directory):
    y = []
    X = []
    trainingFileList = listdir(directory)
    for file in trainingFileList:
        y.append(int(file.split('_')[0]))
        X.append(img2vector(directory + '/' + file))
    return np.array(X), np.array(y)
