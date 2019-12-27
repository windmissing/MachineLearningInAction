import numpy as np

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
    return dataSet, labels

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    loops = 500
    alpha = 0.001
    coeff = np.ones((len(dataMatIn[0]), 1))  # 特征数n * 1

    dataMatrix = np.mat(dataMatIn)  # 样本数m * 特征数n
    labelMatrix = np.mat(classLabels).transpose()  # 样本数m * 1
    for i in range(loops):
        error = sigmoid(dataMatrix * coeff) - labelMatrix
        # 偏导根据上面的公式计算，去掉了常量1/m
        coeff = coeff - alpha * dataMatrix.transpose() * error
    return coeff