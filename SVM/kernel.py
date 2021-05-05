# -*- coding: utf-8 -*-
import numpy as np
import utils

# X: m*n
# y: 1*n向量映射成1*m的向量
def kernelTrans(X, y, kernel, sigma):
    if kernel == product:
        temp = y * X.T
    else:
        k = np.mat(np.zeros((X.shape[0], 1)))
        for j in range(X.shape[0]):
            k[j] = (X[j] - y)*(X[j] - y).T
        temp = np.exp(-k/(sigma**2)).T
    return temp


def product(I, J, sigma):
    return I['x'] * J['x'].T

def gaussianKernel(I, J, sigma):
    return - (J['x'] - I['x']).dot((J['x'] - I['x']).T) / (sigma**2)


