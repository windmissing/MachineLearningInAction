import numpy as np

def loadDataSet(fileName):
    dataMat = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataMat.append(dataList)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[dataSet[:,feature]>value]
    mat1 = dataSet[dataSet[:,feature]<=value]
    return mat0, mat1

def regErr(dataSet):
    return dataSet[:,-1].var() * dataSet.shape[0]

def regLeaf(dataSet):
    return dataSet[:,-1].mean()

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 叶子条件1：dataSet中只有一种y
    if len(set(dataSet[:,-1])) == 1:
        return None, dataSet[0, -1]
    bestErr, bestFeature, bestValue = np.inf, 0, 0
    for feature in range(dataSet.shape[1]-1): # For every feature, exclude label
        for value in set(dataSet[:,feature]): # For every unique value:
            mat0, mat1 = binSplitDataSet(dataSet,feature,value) # Split the dataset it two
            # 如果mat0和mat1包含的data过少，这个split不生效
            # ops[0]为用户定义的最小的叶子的大小
            if mat0.shape[0] < ops[0] or mat1.shape[0] < ops[0]:
                continue
            newErr = errType(mat0) + errType(mat1) # Measure the error of the two splits
            if newErr < bestErr: # If the error is less than bestError
                bestErr, bestFeature, bestValue = newErr, feature, value # set bestSplit to this split and update bestError
    # 如果优化幅度过小，也没有必要再分
    # ops[1]是用户定义的最小的优化幅度
    if (errType(dataSet)-bestErr ) < ops[1]:
        return None, regLeaf(dataSet) # 返回值为y的中值
    return bestFeature, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feature, value = chooseBestSplit(dataSet, leafType, errType, ops)# Find the best feature to split on:
    if feature == None: # If we can’t split the data
        return value    # this node becomes a leaf node
    retTree = {'feature':feature, 'value':value}
    mat0, mat1 = binSplitDataSet(dataSet, feature, value) # Make a binary split of the data
    retTree['left'] = createTree(mat0,ops=ops) # Call createTree() on the left split of the data
    retTree['right'] = createTree(mat1, ops=ops) # Call createTree() on the right split of the data
    return retTree