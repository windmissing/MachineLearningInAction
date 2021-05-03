# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels, loops = 500, alpha = 0.001):
    coeff = np.ones((len(dataMatIn[0]), 1))  # 特征数n * 1

    dataMatrix = np.mat(dataMatIn)  # 样本数m * 特征数n
    labelMatrix = np.mat(classLabels).transpose()  # 样本数m * 1
    for i in range(loops):
        error = sigmoid(dataMatrix * coeff) - labelMatrix
        # 偏导根据上面的公式计算，去掉了常量1/m
        coeff = coeff - alpha * dataMatrix.transpose() * error
    return coeff

def stocGradAscent0(dataMatrix, classLabels, alpha = 0.01):
    coeff = np.ones((len(dataMatrix[0]), 1))  # 特征数n * 1

    for i, data in enumerate(dataMatrix):
        dataArray = np.mat(data)
        error = sigmoid(dataArray * coeff) - classLabels[i]
        coeff = coeff - alpha * dataArray.T * error
    return coeff

def stocGradAscent1(dataMatrix, classLabels, loops=150):
    coeff = np.ones((len(dataMatrix[0]), 1))  # 特征数n * 1
    for j in range(loops):
        for i, index in enumerate(np.random.permutation(len(dataMatrix))):
            alpha = 4 / (1.0+i+j) + 0.01
            dataArray = np.mat(dataMatrix[index])
            error = sigmoid(dataArray * coeff) - classLabels[index]
            coeff = coeff - alpha * dataArray.T * error
    return coeff

def classify(test_X, coeff):
    ret = []
    for x in test_X:
        y = classifyForOneData(x, coeff)
        ret.append(y)
    return np.array(ret)


def classifyForOneData(inX, weights):
    h = sigmoid(np.mat(inX) * np.mat(weights))
    return h > 0.5
