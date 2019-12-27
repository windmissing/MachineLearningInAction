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

import numpy as np
def trainNB1(trainMatrix,trainCategory):
    matrix = np.array(trainMatrix)
    category = np.array(trainCategory)
    p0Num = matrix[category==0].sum(axis=0) + np.ones(len(trainMatrix[0]))
    p0 = p0Num / p0Num.sum()   # 这一次的分母还是很奇怪
    p1Num = matrix[category==1].sum(axis=0) + np.ones(len(trainMatrix[0]))
    p1 = p1Num / p1Num.sum()
    pAbusive = category[category==1].shape[0] / category.shape[0]
    return np.log(p0), np.log(p1), pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = p0Vec[vec2Classify==1].sum() + np.log(1-pClass1)
    p1 = p1Vec[vec2Classify==1].sum() + np.log(pClass1)
    return int(p1 > p0)

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb=trainNB1(trainMat,listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

