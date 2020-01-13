import numpy as np

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    datArr = [list(map(float, line.strip().split(delim))) for line in fr.readlines()]
    return np.array(datArr)

def pca(dataMat, topNfeat=9999999):
    meanValue = dataMat.mean(axis=0)
    # Remove the mean
    dataMat = dataMat - meanValue
    # Compute the covariance matrix
    covMat = np.cov(dataMat, rowvar=0)
    # Find the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(covMat)
    # Sort the eigenvalues from largest to smallest
    eigValInd = eigVals.argsort()
    # Take the top N eigenvectors
    sortedEigVecs = eigVecs[eigValInd]
    sortedEigVecs = sortedEigVecs[:, -1:(-1-topNfeat):-1]   # 第i个特征值对应第i列特征向量，不是第i列行向量！！！
    # Transform the data into the new space created by the top N eigenvectors
    newDataInLowDimen = dataMat.dot(sortedEigVecs)
    newDataInHighDimen = newDataInLowDimen.dot(sortedEigVecs.T) + meanValue
    return  newDataInLowDimen, newDataInHighDimen

def replaceNanWithMean():
    dataMat = loadDataSet('secom.data', ' ')
    for i in range(dataMat.shape[1]):
        # print (dataMat[np.isnan(dataMat[:,i])])
        dataMat[np.isnan(dataMat[:,i]),i] = dataMat[~np.isnan(dataMat[:,i]), i].mean()
    return dataMat

