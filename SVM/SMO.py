# -*- coding: utf-8 -*-
import numpy as np
import utils
import kernel
np.random.seed(666)

class SMO:
    def __init__(self, kernel, sigma, C, toler, maxIter, verbose=0):
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.verbose = verbose
        self.kernel = kernel
        self.sigma = sigma
        
    # If this error is large, then the alpha corresponding to this data instance can be optimized.
    # ???
    def choose(self, I):
        return ((I['y']*I['E'] < - self.toler) and (I['a'] < self.C)) or \
               ((I['y']*I['E'] > self.toler) and (I['a'] > 0))
    
    def calcLH(self, I, J):
        if (I['y'] != J['y']):
            L = max(0, J['a'] - I['a'])
            H = min(self.C, self.C+J['a']-I['a'])
        else:
            L = max(0, I['a'] + J['a'] - self.C)
            H = min(self.C, I['a'] + J['a'])
        if L == H:
            raise UserWarning('L==H')
        return L, H
    
    # i: the index of first alpha
    # m: the total number of alphas
    # choose a random valuse which is not equal to i
    def selectJrand(self, i, m):
        j = i
        while (j==i):
            j = int(np.random.uniform(0, m))
        return j
    
    def calc_eta(self, kernel, I, J):
        eta = 2.0 * kernel(I, J, self.sigma) - kernel(I, I, self.sigma) - kernel(J, J, self.sigma)
        if eta >= 0:
            raise UserWarning('eta>=0')
        return eta
    
    def update_alpha_i(self, I, J):
        I['gap'] = I['y']*J['y']*(-J['gap'])
        I['a'] = I['a'] + I['gap']
        
    def update_alpha_j(self, I, J):
        # eta: the optional amount to change alpha[j]
        eta = self.calc_eta(self.kernel, I, J)
        alphas_j = J['a']-J['y'] *(I['E'] - J['E']) / eta
        # make sure alpha_j is in [0, C]
        L, H = self.calcLH(I, J)   # 此处要用到更新前的J['a']和I['a']
        alphas_j = self.clipAlpha(alphas_j, H, L)
        J['gap'] = alphas_j - J['a']
        J['a'] = alphas_j
        if(abs(J['gap']) < 0.00001):# the float way to compare
            raise UserWarning('J not moving enough')
            
    # 将aj限制在H和L之间
    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj
    
    def isInBound(self, I):
        return not (0 < I['a']) and (I['a'] < self.C)
    
    def calc_b_gap(self, I, J, target, kernel):
        return float(-target['E'] - I['y']*kernel(I, target, self.sigma)*(I['gap']) - J['y']*kernel(J, target, self.sigma)*(J['gap']))

    # 根据alphas计算w
    # ???
    def get_w(self, alphas, dataset, labels):
        alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
        yx = labels.reshape(1, -1).T * np.array([1,1]) *dataset
        w = np.dot(yx.T, alphas)
        return w.T[0]

    def classify(self, test_X):
        ret = []
        for x in test_X:
            ret.append(self.classifyForOneData(x))
        return np.array(ret)
    
    def classifyForOneData(self, x):
        kernelEval = kernel.kernelTrans(self.supportVectors, x, self.kernel, self.sigma)  # 1*m
        predict = kernelEval.dot((np.multiply(self.supportLabels, self.alphas)).T) + self.b
        return np.sign(predict[0,0])
