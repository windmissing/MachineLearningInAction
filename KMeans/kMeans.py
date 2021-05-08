# -*- coding: utf-8 -*-
# +
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)


# -

def distEclud(vecA, vecB):
    return np.sqrt(((vecA - vecB) ** 2).sum())

def randCent(dataSet, k):
    X = np.array(dataSet)
    ret = np.zeros((k, X.shape[1]))
    for i in range(X.shape[1]):
        ret[:, i] = (X[:,i].max() - X[:,i].min()) * np.random.random_sample(k) + X[:,i].min()
        for j in range(k):
            if ret[j,i] > X[:,i].max() or ret[j][i] < X[:,i].min():
                print('X[i].min=', X[i].min(), 'X[i].max=', X[i].max(), 'ret[j,i]', ret[j][i])
                raise UserWarning('randCent has a bug')
    return ret

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    if dataSet.shape[0] == 0:
        raise UserWarning('kMeans(): dataSet is empty')
    # Create k points for starting centroids (often randomly)
    centerId = randCent(dataSet, k)
    iteration = True
    clusterAssment = np.zeros(dataSet.shape[0])
    # While any point has changed cluster assignment
    while iteration:
        iteration = False
        centerError = np.zeros(k)
        # for every point in our dataset:
        for i, data in enumerate(dataSet):
            # calculate the distance between the centroid and point
            dis = ((data - centerId) ** 2).sum(axis=1)
            center = dis.argmin()
            if clusterAssment[i] != center: iteration = True
            # assign the point to the cluster with the lowest distance
            clusterAssment[i] = center
            centerError[center] += dis[center]
        #plotKMeans(dataSet, k, centerId, clusterAssment) # 可以显示过程的效果，用于debug
        # for every cluster
        for i in range(k):
            #calculate the mean of the points in that cluster
            #assign the centroid to the mean
            if dataSet[clusterAssment==i].shape[0]:  # 有可能一开始有个随机中心点离所有点都很远
                centerId[i, :] = dataSet[clusterAssment==i].mean(axis=0)
            else:
                # plotKMeans(dataSet, k, centerId, clusterAssment)
                raise UserWarning('the center is far from every one')
    # centerError在biKmeans()中会有用
    return centerId, clusterAssment, centerError

def biKmeans(dataSet, k, distMeas=distEclud):
    # Start with all the points in one cluster
    centId0 = np.mean(dataSet, axis=0)
    centList =[centId0]
    clusterAssment = np.zeros(dataSet.shape[0])
    centerError = [((dataSet - centId0)**2).sum()]
    # While the number of clusters is less than k
    while(len(centList)<k):
        bestErrorDiff, bestCenter, bestAssment = 0, 0, clusterAssment
        # for every cluster
        for i in range(len(centList)):
            try:
                # measure total error
                totalError1 = centerError[i]
                # perform k-means clustering with k=2 on the given cluster
                myCentroids, myclustAssing, mycenterError = kMeans(dataSet[clusterAssment==i],2)
                # measure total error after k-means has split the cluster in two
                totalError2 = mycenterError.sum()
                #choose the cluster split that gives the lowest error
                if((totalError1-totalError2)>bestErrorDiff):
                    bestErrorDiff, bestCenter, bestAssment, bestCenterId = totalError1-totalError2, i, myclustAssing, myCentroids
            except UserWarning as err:
                print (err)
        # commit this split
        clust = bestAssment.copy() # 以下两句都要基于旧的myclustAssing更新，所以要先保存一下旧的
        bestAssment[clust==0] = bestCenter
        bestAssment[clust==1] = len(centList)
        centList[bestCenter] = bestCenterId[0]
        centList.append(bestCenterId[1])
        centerError[bestCenter] = mycenterError[0]
        centerError.append(mycenterError[1])
        clusterAssment[clusterAssment == bestCenter]= bestAssment
    return centList, clusterAssment
