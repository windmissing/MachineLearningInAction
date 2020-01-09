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
        return None, leafType(dataSet) # 兼容返回向量的情况
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
    print (errType(dataSet)-bestErr, ops[1])
    if (errType(dataSet)-bestErr ) < ops[1]:
        return None, leafType(dataSet) # 返回值为y的中值
    return bestFeature, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feature, value = chooseBestSplit(dataSet, leafType, errType, ops)# Find the best feature to split on:
    if feature == None: # If we can’t split the data
        return value    # this node becomes a leaf node
    retTree = {'feature':feature, 'value':value}
    mat0, mat1 = binSplitDataSet(dataSet, feature, value) # Make a binary split of the data
    retTree['left'] = createTree(mat0,leafType, errType, ops) # Call createTree() on the left split of the data
    retTree['right'] = createTree(mat1, leafType, errType, ops) # Call createTree() on the right split of the data
    return retTree


def isTree(tree):
    return type(tree).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    # 特殊情况，testData中没有数据
    if testData.shape[0] == 0: return getMean(tree)  # ???没有数据了，返回预测值还有意义吗？
    # Split the test data for the given tree:
    mat0, mat1 = binSplitDataSet(testData, tree['feature'], tree['value'])
    # If the either split is a tree: call prune on that split
    if isTree(tree['left']): tree['left'] = prune(tree['left'], mat0)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], mat1)
    if isTree(tree['left']) or isTree(tree['right']):
        return tree
    # Calculate the error without merging
    # 此时左右子树都不是树，而是值，testData的所有样本都预测为这个值
    errorNoMerge = ((mat0 - tree['left']) ** 2).sum() + ((mat1 - tree['right']) ** 2).sum()
    # Calculate the error associated with merging two leaf nodes
    # 两个叶子merge后，新的叶子仍不是一个子树，而是值，是原叶子的平均值
    treeMean = (tree['left'] + tree['right']) / 2
    errorMerge = ((testData - treeMean) ** 2).sum()
    # If merging results in lower error then merge the leaf nodes
    if errorMerge < errorNoMerge:
        print('merging')
        return treeMean
    else:
        return tree