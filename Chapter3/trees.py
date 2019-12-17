from math import log
def calcShannonEnt(dataSet):
    count = {}
    for data in dataSet:
        label = data[-1]
        count[label] = count.get(label, 0) + 1
    sum = len(dataSet)
    entropy = 0.0
    for key in count.keys():
        p_key = count[key] / sum
        entropy = entropy - p_key * log(p_key, 2)    # entropy的计算可以以2为底或者以e为底。本书中是以2为底。
    return entropy

def createDataSet():
    dataSet = [[1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    newDataSet = []
    for data in dataSet:
        if data[axis] == value:     # 如果用numpy，可直接写成dataSet[dataSet[:,axis]==value]
            newDataSet.append(data[:axis] + data[axis+1:])   # 如果用numpy，可写成np.hstack([dataSet[:, :axis], dataSet[:, axis+1])
    return newDataSet

def chooseBestFeatureToSplit(dataSet):
    bestEntropy = 1e8
    bestFeature = 0
    for i in range(len(dataSet[0])-1):
        entropy = 0
        s = set([data[i] for data in dataSet])
        for v in s:
            subDataSet = splitDataSet(dataSet, i, v)
            entropy += len(subDataSet)/len(dataSet) * calcShannonEnt(subDataSet)
        if entropy < bestEntropy:
            bestEntropy = entropy
            bestFeature = i
    return bestFeature