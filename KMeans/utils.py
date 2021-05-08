# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataMat.append(dataList)
    return dataMat

def plotKMeans(dataSet, k, centerId, clusterAssment):
    for i in range(k):
        plt.scatter(dataSet[clusterAssment==i,0], dataSet[clusterAssment==i,1])
        plt.scatter(centerId[i, 0],centerId[i, 1], marker='+', s = 150, c='black')
    plt.show()
