# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataSet = []
    labels = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        data = line.strip().split()
        # 手动增加了一个数据x_0 = 1
        # 读进来的data都是字符串，需要转成数值型
        dataSet.append([1.0, float(data[0]), float(data[1])])
        labels.append(int(data[-1]))
    return np.array(dataSet), np.array(labels)

def plotBestFit(train_X,train_y, coeff):
    # 画点
    plt.scatter(train_X[train_y==0,1], train_X[train_y==0, 2])
    plt.scatter(train_X[train_y==1,1], train_X[train_y==1, 2])
    # 画线
    w = coeff.T[0]
    x = np.array([-3.0, 3.0])
    y = (-w[0]-w[1]*x)/w[2]
    plt.plot(x, y)
    # 画其它
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def loadColicData(filename):
    X = []
    y = []
    for line in open(filename).readlines():
        dataList = [1] + [float(data) for data in line.strip().split('\t')]  # 应该增加一个x0 =1，但书上没有[1]
        X.append(dataList[0:-1])
        y.append(dataList[-1])
    return np.array(X), np.array(y)


def calculateAccuray(predict, target):
    s = 0
    for i in range(predict.shape[0]):
        if predict[i] == target[i]:
            s += 1
    return s/predict.shape[0]
