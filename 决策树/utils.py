# -*- coding: utf-8 -*-
def createDataSet():
    dataSet = [[1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

import operator
def majorityCnt(classList):
    count = {}
    for feature in classList:
        count[feature] = count.get(feature, 0) + 1
    sortedCount = sorted(count.items(),   # iterable -- 可迭代对象，在python2中使用A.iteritems()，在python3中使用A.items()
                           key=operator.itemgetter(1),   # key -- 主要是用来进行比较的元素，指定可迭代对象中的一个元素来进行排序，这里指基于item的value进行排序
                           reverse=True)    # reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    return count[0][0]

# dataSet的最后一个特征是label
# labels数组的第i个字符串代表dataSet第i个特征的含义
def createTree(dataSet, labels):
    # 如果dataSet的标签都是一样的
    labelList = [data[-1] for data in dataSet]  # 假设最后一个数据是标签
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]
    # 如果已经没有feature可以分了
    if len(dataSet[0]) == 1:
        return majorityCnt(labelList)

    myTree = {}
    bestFeature = chooseBestFeatureToSplit(dataSet)
    myTree[labels[bestFeature]] = {}
    values = set([value[bestFeature] for value in dataSet])
    for v in values:
        newLabels = labels[:bestFeature] + labels[bestFeature + 1:]
        newDataSet = splitDataSet(dataSet, bestFeature, v)
        myTree[labels[bestFeature]][v] = createTree(newDataSet, newLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    feature = list(inputTree.keys())[0]
    featureId = featLabels.index(feature)
    value = testVec[featureId]
    subTree = inputTree[feature][value]
    if type(subTree).__name__ == 'dict':
        return classify(subTree, featLabels, testVec)
    return str(subTree)

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
