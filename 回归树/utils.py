# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataMat.append(dataList)
    return np.array(dataMat)

def plotTree(start, end, tree):
    if isTree(tree):
        plotTree(start, tree['value'], tree['right'])
        plotTree(tree['value'], end, tree['left'])
    else:
        x = np.array([start, end])
        y = np.array([[1, start], [1, end]]).dot(tree)
        plt.plot(x, y)

def linearSolver(dataSet):
    xArr = np.array(dataSet)
    X, y = np.hstack([np.ones((xArr.shape[0],1)), xArr[:, 0:-1]]), xArr[:, -1]  # X需要增加一个x0=1
    xTx = X.T.dot(X)
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
                         try increasing the second value of ops')
    ws = np.mat(xTx).I.dot(X.T).dot(y)
    return ws.A[0], X, y  # matrix类型的ws转成array类型的的ws

def modelLeaf(dataSet):
    ws,_,_ = linearSolver(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,y = linearSolver(dataSet)
    yHat = X.dot(ws)
    return ((y-yHat)**2).sum()

def plotModelTree(file):
    data = np.array(loadDataSet('exp2.txt'))
    tree = createTree(data, modelLeaf, modelErr, (1, 10))
    plt.scatter(data[:, 0], data[:, 1], c='gray')
    plotTree(0, 1.0, tree)
    plt.show()

def regTreeEval(tree, data):
    return tree

def modelTreeEval(tree, data):
    X = np.hstack([1, data])
    return X.dot(tree)

def treeForecast(tree, data, modelEval):
    if isTree(tree):
        if data[tree['feature']] > tree['value']:
            return treeForecast(tree['left'], data, modelEval)
        else:
            return treeForecast(tree['right'], data, modelEval)
    else:
        return modelEval(tree, data)

def createForecast(tree, testData, modelEval=regTreeEval):
    predict = []
    for data in testData:
        predict.append(treeForecast(tree, np.array([data]), modelEval))
    return predict
