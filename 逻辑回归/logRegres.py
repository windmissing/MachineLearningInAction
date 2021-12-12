# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
    # return np.exp(inX) / (1 + np.exp(inX))

np.random.seed(666)
def gradAscent(X_train, y_train, lr = 1e-3, loops = 500):
    y_train = y_train.reshape([-1,1])
    W = np.random.random([X_train.shape[1], 1])
    for i in range(loops):
        loss = lossFunction(sigmoid(X_train.dot(W)), y_train)
        if loss < 1e-2:
            break;
        W = W - lr * X_train.T.dot(sigmoid(X_train.dot(W))- y_train)
    return W

def stocGradAscent0(X_train, y_train, lr = 1e-2):
    y_train = y_train.reshape([-1,1])
    W = np.random.random([X_train.shape[1], 1])  # 特征数n * 1

    for i, data in enumerate(X_train):
        data = data.reshape([1,-1])
        loss = lossFunction(sigmoid(data.dot(W)), y_train[i])
        W = W - lr * data.T.dot(sigmoid(data.dot(W))- y_train[i])
    return W

def stocGradAscent1(X_train, y_train, loops=150):
    y_train = y_train.reshape([-1,1])
    W = np.random.random([X_train.shape[1], 1])  # 特征数n * 1
    for j in range(loops):
        for i, index in enumerate(np.random.permutation(len(X_train))):
            alpha = 4 / (1.0+i+j) + 0.01
            data = X_train[index].reshape([1,-1])
            loss = lossFunction(sigmoid(data.dot(W)), y_train[index])
            W = W - alpha * data.T.dot(sigmoid(data.dot(W))- y_train[index])
    return W

# def classify(test_X, coeff):
#     ret = []
#     for x in test_X:
#         y = classifyForOneData(x, coeff)
#         ret.append(y)
#     return np.array(ret)

def classify(W, test_X):
    y_hat = sigmoid(test_X.dot(W))
    return (y_hat>0.5)

def lossFunction(y_predict, y_target):
    return np.sum(- y_target * np.log(y_predict) - (1-y_target)* np.log(1-y_predict))

# def classifyForOneData(inX, weights):
#     h = sigmoid(np.mat(inX) * np.mat(weights))
#     return h > 0.5
