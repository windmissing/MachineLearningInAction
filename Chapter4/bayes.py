def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
    'problems', 'help', 'please'],
    ['maybe', 'not', 'take', 'him', \
    'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', \
    'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
    'to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set()
    for data in dataSet:
        vocabSet = vocabSet | set(data)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    vec = [0] * len(vocabList)
    for word in inputSet:
        vec [vocabList.index(word)] = 1
    return vec

import numpy as np
def trainNB0(trainMatrix,trainCategory):
    matrix = np.array(trainMatrix)
    category = np.array(trainCategory)
    p0Num = matrix[category==0].sum(axis=0)
    p0 = p0Num / p0Num.sum()   # 为什么是p0Num.sum()？不是category[category==0].shape[0]？
    p1Num = matrix[category==1].sum(axis=0)
    p1 = p1Num / p1Num.sum()
    pAbusive = category[category==1].shape[0] / category.shape[0]
    return p0, p1, pAbusive