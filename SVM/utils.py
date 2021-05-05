# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataSet = []
    labels = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataSet.append(dataList[0:-1])
        labels.append(dataList[-1])
    return np.array(dataSet), np.array(labels)

def showSMO(dataArr, labelArr, alphas, w, b):
    dataMat, labelMat = np.array(dataArr), np.array(labelArr)

    # 绘制样本点
    plt.scatter(dataMat[labelMat==1,0], dataMat[labelMat==1,1])
    plt.scatter(dataMat[labelMat==-1,0], dataMat[labelMat==-1,1])

    # 绘制决策边界
    x = np.array([0, 10])
    y = (-b - x*w[0])/w[1]
    plt.plot(x, y)

    # 绘制支撑向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x,y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='#AB3319')
    plt.show()

def calculateAccuray(predict, target):
    s = 0
    for i in range(predict.shape[0]):
        if predict[i] == target[i]:
            s += 1
    return s/predict.shape[0]


def img2vector(filename):
    ret = np.zeros((0))
    fr = open(filename)
    for line in fr.readlines():
        line = line[:32]
        newinfo = np.array(list(line), dtype=int)
        ret = np.hstack([ret, newinfo])
    return ret.reshape(1,-1)

from os import listdir
def loadImages(dirName):
    labels = []
    dataSet = np.zeros((0,1024))
    trainingFileList = listdir('digits/'+dirName)
    for file in trainingFileList:
        digit = int(file.split('_')[0])
        if digit == 9:
            labels.append(-1)
        else:
            labels.append(1)
        dataSet = np.vstack([dataSet, img2vector('digits/'+dirName+'/'+file)])
    return dataSet, labels
