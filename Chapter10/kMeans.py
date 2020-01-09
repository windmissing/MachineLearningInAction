import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataMat.append(dataList)
    return dataMat

def distEclud(vecA, vecB):
    return np.sqrt(((vecA - vecB) ** 2).sum())

def randCent(dataSet, k):
    X = np.array(dataSet)
    ret = np.zeros((k, X.shape[1]))
    for i in range(X.shape[1]):
        ret[:, i] = np.random.randint(X[i].min(), X[i].max(),k)
    return ret

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # Create k points for starting centroids (often randomly)
    centerId = randCent(dataSet, k)
    iteration = True
    clusterAssment = np.zeros(dataSet.shape[0])
    # While any point has changed cluster assignment
    while iteration:
        iteration = False;
        # for every point in our dataset:
        for i, data in enumerate(dataSet):
            # calculate the distance between the centroid and point
            dis = ((data - centerId) ** 2).sum(axis=1)
            center = dis.argmin()
            if clusterAssment[i] != center: iteration = True
            # assign the point to the cluster with the lowest distance
            clusterAssment[i] = center
        # for every cluster
        for i in range(k):
            #calculate the mean of the points in that cluster
            #assign the centroid to the mean
            if dataSet[clusterAssment==i].shape[0]:  # 有可能一开始有个随机中心点离所有点都很远
                centerId[i, :] = dataSet[clusterAssment==i].mean(axis=0)
    return centerId, clusterAssment

def plotKMeans(dataSet, k, centerId, clusterAssment):
    for i in range(k):
        plt.scatter(dataSet[clusterAssment==i,0], dataSet[clusterAssment==i,1])
        plt.scatter(centerId[i, 0],centerId[i, 1], marker='+', s = 150, c='black')
    plt.show()