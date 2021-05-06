# -*- coding: utf-8 -*-
import numpy as np

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


# performs a threshold comparison to classify data
# Everything on one side of the threshold is thrown into class -1, and everything on the other side is thrown into class +1
def stumpClassify(dataMatrix, feature, value, ineq):
    retArray = np.ones(dataMatrix.shape[0])
    if ineq == 'lt':
        retArray[dataMatrix[:,feature] <= value] = -1
    else:
        retArray[dataMatrix[:,feature] > value] = -1
    return retArray

# week classifier
# iterate over all of the possible inputs, find the best decision stump
# Best here will be with respect to the data weight vector  D
def buildStump(dataArr, classLabels, D):
    X, y = np.array(dataArr), np.array(classLabels)
    m,n = np.shape(X);numSteps = 10
    bestStump, bestClasEst, minError = {}, np.zeros(m), np.inf # Set the minError to +
    for i in range(n):  # For every feature in the dataset
        valueRange = (X[:, i].min(), X[:, i].max())   # value的遍历范围
        stepSize = (valueRange[1]-valueRange[0])/numSteps
        for step in range (-1, numSteps+1): # numSteps+2个不同的value取值  ？？？超出边界的分类有什么必要？
            for ineq in ['lt', 'gt']: # 左边为1右边为-1，或者左边为-1右边为1
                value = valueRange[0] + step * stepSize  # ???
                predict = stumpClassify(X, i, value, ineq) # Build a decision stump
                err = np.ones(m)
                err[predict == y] = 0
                weightedError = D.dot(err.T) # test it with the weighted dataset
                #print ("split: feature %d, value %.2f, ineq: %s, the weighted error is %.3f" % (i, value, ineq, weightedError))
                # If the error is less than minError
                if weightedError < minError:
                    # set this stump as the best stump
                    bestStump, bestClasEst, minError = {'feature':i, 'value':value, 'ineq':ineq}, predict.copy(), weightedError
    return bestStump, bestClasEst, minError

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weekClassifier = []
    m = len(classLabels)
    D = np.ones(m)/m # 初始化时所有weight都是相等的
    # aggPredict迭代到当前状态时，对训练样本的分类的综合评估
    # 估计结果以正负代表分类，具体的值不一定是1或者-1
    # 初始化为0，即无任何分类倾向
    aggPredict = np.zeros(m)
    for i in range(numIt): # For each iteration:
        stump, predict, error = buildStump(dataArr,classLabels,D) # Find the best stump using buildStump()
        # Calculate alpha
        # note 1: 公式中是ln，怎么代码里变成log了？
        # note 2: 分母写成max(error,1e-16)是为了避免下溢错误
        # note 3: error越大，alpha越小，训练结果越不可信，D和appPreidct更新越少
        print ('D=',D)
        stump['alpha'] = 0.5 * np.log((1-error)/max(error,1e-16))
        print ("predict: ", predict)
        weekClassifier.append(stump) # Add the best stump to the stump array
        # Calculate the new weight vector – D
        # 当预测错误时，y_predict * y_true = -1, 当预测正确时，y_predict * y_true = 1
        # 因此exp的指数为-y_preict * y_true * alpha
        expon = -predict * np.array(classLabels) * stump['alpha']
        D = D * np.exp(expon);D /=  D.sum() # 这两步要分开写，因为第二步用的D是新的D。第二步是为了保证D.sum()始终为1
        aggPredict += stump['alpha'] * predict # Update the aggregate class estimate
        print ('aggpredict=',aggPredict)
        # 正负号代表预测结果，符号相同即预测正确
        # note 1: errorRate和error不同的是，errorRate不考虑权重，只是错误样本所占的比例
        aggErrorRate = np.sum(np.sign(aggPredict) != np.array(classLabels))/ m
        print ("total error: ",aggErrorRate)
        if aggErrorRate == 0.0:break
    return weekClassifier, aggPredict

def adaClassify(datToClass,classifierArr):
    data = np.array(datToClass)
    ret = np.zeros(data.shape[0])
    for classifier in classifierArr:
        predict = stumpClassify(data, classifier['feature'], classifier['value'], classifier['ineq'])
        predict = predict * classifier['alpha']
        ret += predict
    print (ret)
    return np.sign(ret)

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        dataList = [1] + [float(data) for data in line.strip().split('\t')]  # 应该增加一个x0 =1，但书上没有[1]
        dataMat.append(dataList[0:-1])
        labelMat.append(dataList[-1])
    return dataMat, labelMat
