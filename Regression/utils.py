# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataMat.append(dataList[0:-1])
        labelMat.append(dataList[-1])
    return np.array(dataMat), np.array(labelMat)

def plotLwlrTest(xArr, yArr, predictList):
    sortedIndex = xArr[:,1].argsort()
    plt.scatter(xArr[:,1], yArr, s=2)
    for i, predict in enumerate(predictList):
        plt.plot(xArr[sortedIndex,1],predict[sortedIndex],  label=str(i))
    plt.legend()
    plt.show()

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()


def ridgeRegres(xArr,yArr,lam=0.2):
    X, y = np.array(xArr), np.array(yArr)
    xTx = X.T.dot(X) + lam * np.eye(X.shape[1])
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = np.mat(xTx).I.dot(X.T).dot(y)
    return ws.T

def plotRidge():
    abX,abY=loadDataSet('abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)
    import matplotlib.pyplot as plt
    print (ridgeWeights.shape)
    plt.plot(ridgeWeights)
    plt.show()

# 标准化
# 这个作者对标准化（Standardization）、归一化（normalization）、正则化（regularization）这三个术语有什么误解？
# 一会用normalization一会又用regularization。但它实际上做的是类似于Standardization的事情。
# 但也只是类似，Standardization公式的分母是标准差，但他用的又是方差。非常奇怪。
# 另外，对y做预处理似乎也没有意义
def standardization(dataSet):
    mean = np.mean(dataSet, axis=0)
    std = np.std(dataSet, axis=0)
    return ((dataSet - mean) / std)

def ridgeTest(xArr,yArr):
    X = standardization(np.array(xArr))
    y = np.array(yArr)
    ws = np.zeros((0, X.shape[1]))
    for i in range(-10, 20):
        w = ridgeRegres(X,y,np.exp(i)).T
        ws = np.vstack([ws, w])
    return ws

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    # Regularize the data to have 0 mean and unit variance
    X = standardization(np.array(xArr))
    y = np.array(yArr)
    ws = np.zeros(X.shape[1])
    for iter in range(numIt): # For every iteration:
        lowestError = np.inf #Set lowestError to +inf
        for feature in range(X.shape[1]): # For every feature:
            for sign in [-1, 1]: # For increasing and decreasing:
                ws_local = ws.copy()
                ws_local[feature] += eps * sign # Change one coefficient to get a new W
                err = rssError(y, X.dot(ws_local.T)) # Calculate the Error with new W
                if err < lowestError: # If the Error is lower than lowestError:
                    lowestError = err
                    ws_local_best = ws_local.copy() # set Wbest to the current W
        ws = ws_local_best.copy() # Update set W to Wbest
    return ws

def plotStageWise():
    xArr,yArr=loadDataSet('abalone.txt')
    testNum = 5000
    ws = np.zeros((testNum, len(xArr[0])))
    for i in range(testNum):
        ws[i,:] = stageWise(xArr,yArr,0.01,i)
    plt.plot(ws)
    plt.show()
