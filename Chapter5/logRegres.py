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

import numpy as np
import matplotlib.pyplot as plt

def plotBestFit(func):
    dataArr,labelMat=loadDataSet()
    # 画点
    data = np.array(dataArr)
    label = np.array(labelMat)
    plt.scatter(data[label==0,1], data[label==0, 2])
    plt.scatter(data[label==1,1], data[label==1, 2])
    # 画线
    weight = func(dataArr,labelMat)
    w = weight.T.getA()[0]
    x = np.array([-3.0, 3.0])
    y = (-w[0]-w[1]*x)/w[2]
    plt.plot(x, y)
    # 画其它
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    alpha = 0.01
    coeff = np.ones((len(dataMatrix[0]), 1))  # 特征数n * 1

    for i, data in enumerate(dataMatrix):
        dataArray = np.mat(data)
        error = sigmoid(dataArray * coeff) - classLabels[i]
        coeff = coeff - alpha * dataArray.T * error
    return coeff

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    coeff = np.ones((len(dataMatrix[0]), 1))  # 特征数n * 1
    for j in range(numIter):
        for i, index in enumerate(np.random.permutation(len(dataMatrix))):
            alpha = 4 / (1.0+i+j) + 0.01
            dataArray = np.mat(dataMatrix[index])
            error = sigmoid(dataArray * coeff) - classLabels[index]
            coeff = coeff - alpha * dataArray.T * error
    return coeff

def classifyVector(inX, weights):
    h = sigmoid(np.mat(inX) * np.mat(weights))
    return h > 0.5


def colicTest():
    trainData = []
    trainLabel = []
    for line in open('horseColicTraining.txt').readlines():
        dataList = [1] + [float(data) for data in line.strip().split('\t')]  # 应该增加一个x0 =1，但书上没有[1]
        trainData.append(dataList[0:-1])
        trainLabel.append(dataList[-1])
    trainWeights = stocGradAscent1(trainData, trainLabel, 500)

    error, total = 0, 0
    for line in open('horseColicTest.txt').readlines():
        total += 1
        dataList = [1] + [float(data) for data in line.strip().split('\t')]
        testData = dataList[0:-1]
        testLabel = dataList[-1]
        y = classifyVector(testData, trainWeights)
        if y != testLabel:
            error += 1
    return error / total

def multiTest():
    numTests=10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()   # 这种写法效率很低，读文件很耗时且重复多次没有意义
    return errorSum / numTests