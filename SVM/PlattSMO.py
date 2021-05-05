# -*- coding: utf-8 -*-
import numpy as np
from SMO import SMO
import kernel
class optStruct:
    def __init__(self, dataMatIn, classLabels):
        self.X = np.mat(dataMatIn)
        self.labelMat = np.mat(classLabels).transpose()
        self.m= np.shape(self.X)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))# 把所有alpha都初始化为0
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) # m*2, [i,0]代表[i,1]值是否有效。[i,1]代表E_i


class PlattSMO(SMO):
    def __init__(self, C, toler, maxIter, verbose=0, kernel=kernel.product, sigma=0.1):
        super(PlattSMO, self).__init__(kernel, sigma, C, toler, maxIter, verbose)
        
    def train(self, dataMatIn, classLabels):
        oS = optStruct(dataMatIn, classLabels)
        iter = 0
        entireSet, alphaPairsChanged = True, 0
        while (iter < self.maxIter) and ((alphaPairsChanged > 0) or (entireSet == True)):
            alphaPairsChanged = 0
            iter += 1
            l = np.arange(oS.m)
            if not entireSet:
                l = l[np.array(oS.alphas.A[:, 0] > 0) & np.array(oS.alphas.A[:, 0] < self.C)]  # oS.alphas.A[:,0]为什么是list不是array?
            for i in l:
                alphaPairsChanged += self.innerL(i, oS)
            if self.verbose:
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            if entireSet == True:
                entireSet = False
            elif alphaPairsChanged == 0:
                if entireSet == False:
                    entireSet = True
        self.oS = oS
        return oS
    
    def innerL(self, i, oS):
        try:
            I = self.generateI(oS, i)
            J = self.generateJ(oS, I)
            self.update(oS, I, J)
        except UserWarning as err:
            if self.verbose:
                print (err)
            return 0
        return 1
    
    def generateI(self, oS, i):
        # 如果这一轮要更新alpha_i，先把alpha_i对应的样本信息都准备好
        I = self.generateNode(oS, i)
        # condition to choose i: error is big enough
        if (not self.choose(I)):
            raise UserWarning('error not big enough')
        return I
    
    def generateJ(self, oS, I):
        # select j randomly
        j, _ = self.selectJ(I['index'], oS, I['E'])
        return self.generateNode(oS, j)

    def generateNode(self, oS, index):
        return {'index': index,  # 更新第几个alpha
                'x': oS.X[index, :],  # 对应的数据
                'y': oS.labelMat[index],  # 对应的标签
                'a': oS.alphas[index],  # 当前的alpha值
                # E: error between predition class and real class
                # 基于当前alpha对第index个数据的预测分类与该数据的真实分类做比较
                'E': self.calcEk(oS, index)
                }

    # 相当于上面的def E(dataMat, labelMat, alphas, i, b)
    def calcEk(self, oS, i):
        # fx: the predition of the class
        temp = kernel.kernelTrans(oS.X, oS.X[i], self.kernel, self.sigma)
        fXi = float(temp * np.multiply(oS.alphas, oS.labelMat)) + oS.b
        # E: error between predition class and real class
        Ei = fXi - oS.labelMat[i]
        return Ei
    
    # 更新alpha_i, alpha_j和b，使得目标函数进一步变大
    def update(self, oS, I, J):
        # 根据公式先更新alpha_j
        self.update_alpha_j(I, J)
        self.updateEk(oS, J)
        # change alpha_i as alpha_j changed
        self.update_alpha_i(I, J)
        self.updateEk(oS, I)
        # 当更新了一对a_i,a_j之后，需要重新计算b。
        oS.b = self.update_b(oS, I, J)
        # return I['a'], J['a'], oS.b

    def update_b(self, oS, I, J):
        b1 = oS.b + self.calc_b_gap(I, J, I, self.kernel)
        b2 = oS.b + self.calc_b_gap(I, J, J, self.kernel)
        if (not self.isInBound(I)):
            b = b1
        elif (not self.isInBound(J)):
            b = b2
        else:
            b = (b1 + b2)/2.0
        return b


    # 类似于前面的def selectJrand(i, m):
    def selectJ(self, i, oS, Ei):
        maxj,maxEj,maxDeltaE = 0, 0, -1
        oS.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(oS.eCache.A[:, 0])
        if len(validEcacheList) > 1: # len至少为1，因为至少eCache[i]=1。但如果len=1，说明这是第一次运行，只有eCache[i]是有效的。
            for j in validEcacheList:
                Ej = self.calcEk(oS, j)
                # choose the second alpha so that we’ll take the maximum step during each optimization
                if abs(Ej-Ei) > maxEj:  #
                    maxk, maxEj, maxDeltaE = j, Ej, abs(Ej-Ei)
        else:
            maxj = self.selectJrand(i, oS.m)
            maxEj = self.calcEk(oS, maxj)
        return maxj, maxEj

    def updateEk(self, oS, K):
        k = K['index']
        oS.alphas[k] = K['a']
        oS.eCache[k] = [1, self.calcEk(oS, k)]
        
    def collectCoeff(self, oS):
        index = oS.alphas.T.A[0] > 1e-3
        self.b = oS.b
        self.alphas = oS.alphas[index].T.A[0]
        self.supportVectors = oS.X[index]
        self.supportLabels = oS.labelMat[index].T.A[0]
        if self.verbose:
            print("there are %d Support Vectors" % np.sum(index))
            print ("b = ", self.b)
            print ("alphas = ", self.alphas)
            print ("support vectors = ", self.supportVectors)
            print ("support vectors labels = ", self.supportLabels, self.supportLabels.shape)
        # self.w = self.get_w(self.alphas, oS.X, oS.labelMat)
        # print ("w = ", self.w)        


