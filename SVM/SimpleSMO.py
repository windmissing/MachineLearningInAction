# -*- coding: utf-8 -*-
import numpy as np
from SMO import SMO
import kernel


class SimpleSMO(SMO):
    def __init__(self, C, toler, maxIter, verbose=0, kernel=kernel.product, sigma=0.1):
        super(SimpleSMO, self).__init__(kernel, sigma, C, toler, maxIter, verbose)
        
    # 使用SMO算法求出alpha，然后根据alpha计算w和b
    def train(self, dataMatIn, classLabels):
        dataMat, labelMat = np.mat(dataMatIn), np.mat(classLabels).transpose()
        m= np.shape(dataMat)[0]
        alphas = np.mat(np.zeros((m, 1)))# 把所有alpha都初始化为0
        b, iter = 0, 0
        # 迭代matIter次
        while(iter < self.maxIter):
            alphaPairsChanged = 0
            for i in range(m):
                try: # 可能会有一些情况导致选择的这一对ahpha不能更新
                    # 选择一个要更新的alpha
                    I = self.generateI(dataMat, labelMat, alphas, i, b)
                    # 选择另一个要更新的alpha
                    J = self.generateJ(dataMat, labelMat, alphas, i, b, m)
                    # 更新alpha_i, alpha_j，b也需要相应的更新
                    alphas[i], alphas[J['index']], b = self.update(I, J, b)
                    alphaPairsChanged += 1
                    if self.verbose:
                        print ("iter: %d, i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
                except UserWarning as err:
                    if self.verbose:
                        print (err)
            if alphaPairsChanged == 0:
                iter+=1
            else:
                iter = 0
            if self.verbose:
                print("iteration number: %d" % iter)
        return b, alphas
    
    def generateI(self, dataMat, labelMat, alphas, i, b):
        # 如果这一轮要更新alpha_i，先把alpha_i对应的样本信息都准备好
        I = self.generateNode(dataMat, labelMat, alphas, i, b)
        # condition to choose i: error is big enough
        if(not self.choose(I)):
            raise UserWarning('error not big enough')
        return I
    
    def generateJ(self, dataMat, labelMat, alphas, i, b, m):
        # select j randomly
        j = self.selectJrand(i, m)
        return self.generateNode(dataMat, labelMat, alphas, j, b)
    
    def generateNode(self, dataMat, labelMat, alphas, index, b):
        return {'index':index, # 更新第几个alpha
                 'x':dataMat[index,:], # 对应的数据
                 'y':labelMat[index],     # 对应的标签
                 'a':alphas[index],       # 当前的alpha值
                #E: error between predition class and real class
                # 基于当前alpha对第index个数据的预测分类与该数据的真实分类做比较
                 'E':self.E(dataMat, labelMat, alphas, index, b)
               }
    
    def E(self, dataMat, labelMat, alphas, i, b):
        # fx: the predition of the class
        fXi = float(np.multiply(alphas, labelMat).T*(dataMat*dataMat[i,:].T)) + b
        # E: error between predition class and real class
        Ei = fXi - labelMat[i]
        return Ei
    
    # 更新alpha_i, alpha_j和b，使得目标函数进一步变大
    def update(self, I, J, b):
        # 根据公式先更新alpha_j
        self.update_alpha_j(I, J)
        # change alpha_i as alpha_j changed
        self.update_alpha_i(I, J)
        # 当更新了一对a_i,a_j之后，需要重新计算b。
        b = self.update_b(I, J, b)
        return I['a'], J['a'], b
    
    def update_b(self, I, J, b):
        b1 = b + self.calc_b_gap(I, J, I, self.kernel)
        b2 = b + self.calc_b_gap(I, J, J, self.kernel)
        if (not self.isInBound(I)):
            b = b1
        elif (not self.isInBound(J)):
            b = b2
        else:
            b = (b1 + b2)/2.0
        return b
