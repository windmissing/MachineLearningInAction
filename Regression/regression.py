# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import utils

class Regression():
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

class StandRegre(Regression):
    """直接用公式计算线性回归的闭式解"""
    def __init__(self, train_X, train_y, test_X, test_y):
        super(StandRegre, self).__init__(train_X, train_y, test_X, test_y)
        
    def train(self):
        xTx = self.train_X.T.dot(self.train_X)
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = np.mat(xTx).I.dot(self.train_X.T).dot(self.train_y)
        self.ws = ws.T.A
        return self.ws
    
    def predict(self):
        return self.test_X.dot(self.ws)[:,0]

class Lwlr(Regression):
    """局部加权线性回归，locally weighted linear regression"""
    def __init__(self, train_X, train_y, test_X, test_y):
        super(Lwlr, self).__init__(train_X, train_y, test_X, test_y)
        self.m = self.train_X.shape[0]
        
    def train_and_predict(self, k):
        predict = []
        for test in self.test_X:
            self.train_for_one_data(test, k)
            predict.append(self.predict_for_one_data(test))
        return np.array(predict)
    
    def train_for_one_data(self, test, k):
        # X, y, test = np.array(xArr), np.array(yArr), np.array(testPoint)
        # m = X.shape[0]
        W = np.zeros((self.m,self.m))
        for i in range(self.m):
            diffMat = self.train_X[i] - test
            W[i,i] = np.exp(diffMat.dot(diffMat.T)/(-2*k**2))
        xTwx = self.train_X.T.dot(W).dot(self.train_X)
        if np.linalg.det(xTwx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        self.w = np.mat(xTwx).I.dot(self.train_X.T).dot(W).dot(self.train_y)
    
    def predict_for_one_data(self, test):
        return test.dot(self.w.T)[0,0]

class RidgeRegre(Regression):
    """直接用公式计算线性回归的闭式解"""
    def __init__(self, train_X, train_y, test_X, test_y):
        super(RidgeRegre, self).__init__(train_X, train_y, test_X, test_y)
        
    def train(self, lam=0.2):
        xTx = self.train_X.T.dot(self.train_X) + lam * np.eye(self.train_X.shape[1])
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = np.mat(xTx).I.dot(self.train_X.T).dot(self.train_y)
        self.ws = ws.T.A
        return self.ws
    
    def predict(self):
        return self.test_X.dot(self.ws)[:,0]


class StageWise(Regression):
    """直接用公式计算线性回归的闭式解"""
    def __init__(self, train_X, train_y, test_X, test_y):
        super(StageWise, self).__init__(train_X, train_y, test_X, test_y)
        
    def train(self, eps=0.01,numIt=100):
        ws = np.zeros(self.train_X.shape[1])
        for iter in range(numIt): # For every iteration:
            lowestError = np.inf #Set lowestError to +inf
            for feature in range(self.train_X.shape[1]): # For every feature:
                for sign in [-1, 1]: # For increasing and decreasing:
                    ws_local = ws.copy()
                    ws_local[feature] += eps * sign # Change one coefficient to get a new W
                    err = utils.rssError(self.train_y, self.train_X.dot(ws_local.T)) # Calculate the Error with new W
                    if err < lowestError: # If the Error is lower than lowestError:
                        lowestError = err
                        ws_local_best = ws_local.copy() # set Wbest to the current W
            ws = ws_local_best.copy() # Update set W to Wbest
            return ws
    
    def predict(self):
        return self.test_X.dot(self.ws)[:,0]


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
                err = rssError(self.train_y, X.dot(ws_local.T)) # Calculate the Error with new W
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
