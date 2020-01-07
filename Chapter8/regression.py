import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataMat.append(dataList[0:-1])
        labelMat.append(dataList[-1])
    return dataMat, labelMat

def standRegres(xArr,yArr):
    X, y = np.array(xArr), np.array(yArr)
    xTx = X.T.dot(X)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = np.mat(xTx).I.dot(X.T).dot(y)
    return ws.T

def lwlr(testPoint,xArr,yArr,k):
    X, y, test = np.array(xArr), np.array(yArr), np.array(testPoint)
    m = X.shape[0]
    W = np.zeros((m,m))
    for i in range(m):
        diffMat = X[i] - testPoint
        W[i,i] = np.exp(diffMat.dot(diffMat.T)/(-2*k**2))
    xTwx = X.T.dot(W).dot(X)
    if np.linalg.det(xTwx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    w = np.mat(xTwx).I.dot(X.T).dot(W).dot(y)
    return test.dot(w.T)[0,0]

def lwlrTest(testArr,xArr,yArr,k):
    m = np.array(xArr).shape[0]
    predict = np.zeros(m)
    for i, test in enumerate(testArr):
        predict[i] = lwlr(test, xArr, yArr, k)
    return predict

def plotLwlrTest(k):
    xArr,yArr=loadDataSet('ex0.txt')
    X, y = np.array(xArr), np.array(yArr)
    yHat = lwlrTest(X, X, y, k)
    sortedIndex = X[:,1].argsort()
    plt.scatter(X[:,1], y, s=2)
    plt.plot(X[sortedIndex,1],yHat[sortedIndex])
    plt.show()

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()