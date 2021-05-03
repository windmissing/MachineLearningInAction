# -*- coding: utf-8 -*-
import numpy as np
def trainNB0(trainMatrix,trainCategory):
    matrix = np.array(trainMatrix)
    category = np.array(trainCategory)
    p0Num = matrix[category==0].sum(axis=0)
    p0 = p0Num / p0Num.sum()   # 为什么是p0Num.sum()？不是category[category==0].shape[0]？
    p1Num = matrix[category==1].sum(axis=0)
    p1 = p1Num / p1Num.sum()
    pAbusive = category[category==1].shape[0] / category.shape[0]
    return p0, p1, pAbusive

import numpy as np
def trainNB1(trainMatrix,trainCategory):
    matrix = np.array(trainMatrix)
    category = np.array(trainCategory)
    p0Num = matrix[category==0, :].sum(axis=0) + np.ones(len(trainMatrix[0]))
    p0 = p0Num / p0Num.sum()   # 这一次的分母还是很奇怪
    p1Num = matrix[category==1, :].sum(axis=0) + np.ones(len(trainMatrix[0]))
    p1 = p1Num / p1Num.sum()
    pAbusive = category[category==1].shape[0] / category.shape[0]
    return np.log(p0), np.log(p1), pAbusive

def classifyNBForOneData(vec2Classify, p0Vec, p1Vec, pClass1):
    # 测试样本vec2Classify属于class 0的概率
    p0 = p0Vec[vec2Classify==1].sum() + np.log(1-pClass1)
    # 测试样本vec2Classify属于class 1的概率
    p1 = p1Vec[vec2Classify==1].sum() + np.log(pClass1)
    return int(p1 > p0)

def classifyNB(test_X, p0Vec, p1Vec, pClass1):
    predict_y = []
    for x in test_X:
        y = classifyNBForOneData(x, p0Vec, p1Vec, pClass1)
        predict_y.append(y)
    return np.array(predict_y)
