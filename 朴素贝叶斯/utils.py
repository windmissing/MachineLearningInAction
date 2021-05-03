# -*- coding: utf-8 -*-
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

def Posts2Mat(myVocabList, listOPosts):
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    return trainMat


def bagOfWords2VecMN(vocabList, inputSet):
    vec = [0] * len(vocabList)
    for word in inputSet:
        vec [vocabList.index(word)] += 1
    return vec


import re
def preprocess(mySent):
    # mySent.split() 这种方法会保留标点符号，因此不能这种方法
    listOfTokens = re.split("\W+", mySent)
    ret = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    return ret


def calculateAccuray(predict, target):
    s = 0
    for i in range(predict.shape[0]):
        if predict[i] == target[i]:
            s += 1
    return s/predict.shape[0]


import numpy as np
def loadSpamData():
    # 读入原始数据，分割成单词的list
    dataSet = []
    for i in range(1,26):
        data = preprocess(open('email/spam/'+str(i)+'.txt').read())
        dataSet.append(data)
    for i in range(1,26):
        data = preprocess(open('email/ham/'+str(i)+'.txt',encoding='ISO-8859-15').read())
        dataSet.append(data)
    labels = np.hstack([np.zeros((26)), np.ones((26))])
    return dataSet, labels


def splitTrainAndTest(dataMat, labels):
    # 随机取40个数据为训练数据，10个数据为测试数据
    index = np.random.permutation(50)
    train_X = np.array(dataMat)[index[10:]]
    train_y = np.array(labels)[index[10:]]
    test_X = np.array(dataMat)[index[:10]]
    test_y = np.array(labels)[index[:10]]
    return train_X,train_y,test_X,test_y
