import numpy as np
from math import sqrt
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# For every point in our dataset:
#     calculate the distance between inX and the current point
#     sort the distances in increasing order
#     take k items with lowest distances to inX
#     find the majority class among these items
#     return the majority class as our prediction for the class of inX
def classify0(inX, dataSet, labels, k):
    # inX：the data to be predicted
    # dataSet：训练集
    # labels：训练集对应的标签
    #datasetSize = dataSet.shape[0]   # 二维数组，行数=array.shape[0]，列数=array.shape[1]
    diff = dataSet - inX
    distance = np.sum(diff**2, axis=1) **0.5
    distanceIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        label = labels[distanceIndex[i]]
        # classCount[label] = classCount[label] + 1   # 如果用字典里没有的键访问数据，会输出错误
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)

    returnMat = np.zeros((numberOfLines, 3))  # 初始化样本特征矩阵
    classLabelVector = []  # 初始化化输出标记列表
    index = 0

    for line in lines:  # 依次处理每一行
        line = line.strip()  # 去掉行尾的换行符
        listFromLine = line.split('\t')  # 提取每个特征
        returnMat[index, :] = listFromLine[0:-1]  # 提取特征向量，并加入到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 提取输出标记，并加入输出列表中

        index += 1
    return returnMat, classLabelVector

def datingClassTest(hoRatio):
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    numTestCase = int(normMat.shape[0] * hoRatio)
    correct = 0
    for i in range(numTestCase):
        result = classify0(normMat[i], normMat[numTestCase:], datingLabels[numTestCase:], 3)
        if result == datingLabels[i]:
            correct = correct + 1
    return correct / numTestCase

def autoNorm(dataSet):
    max = np.max(dataSet, axis=0)
    min = np.min(dataSet, axis=0)
    ranges = max-min
    return ((dataSet -min) / ranges), ranges, min

def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input(\
    "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,  # 对测试用例标准化
                                 normMat,                   # 使用标准化之后的样本
                                 datingLabels, 3)
    print ("You will probably like this person: ",\
    resultList[classifierResult - 1])

def img2vector(filename):
    ret = np.zeros((0))
    fr = open(filename)
    for line in fr.readlines():
        line = line[:32]
        newinfo = np.array(list(line), dtype=int)
        ret = np.hstack([ret, newinfo])
    return ret.reshape(1,-1)

from os import listdir
def downloadTrainingData():
    labels = []
    dataSet = np.zeros((0,1024))
    trainingFileList = listdir("trainingDigits")
    for file in trainingFileList:
        labels.append(int(file.split('_')[0]))
        dataSet = np.vstack([dataSet, img2vector('trainingDigits/'+file)])
    return dataSet, labels

def handwritingClassTest(dataSet, labels):
    error = 0
    total = 0
    testFileList = listdir("testDigits")
    for file in testFileList:
        actualLabel = int(file.split('_')[0])
        expectLabel = classify0(img2vector('testDigits/'+file), dataSet, labels, 3)
        print ("actualLabel = " + str(actualLabel) + ", expectLabel = " + str(expectLabel))
        total += 1
        if actualLabel != expectLabel: error+=1
    return error / total