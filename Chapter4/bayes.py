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
    p0Num = matrix[category==0, :].sum(axis=0) + np.ones(len(trainMatrix[0]))
    p0 = p0Num / p0Num.sum()   # 这一次的分母还是很奇怪
    p1Num = matrix[category==1, :].sum(axis=0) + np.ones(len(trainMatrix[0]))
    p1 = p1Num / p1Num.sum()
    pAbusive = category[category==1].shape[0] / category.shape[0]
    return np.log(p0), np.log(p1), pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 测试样本vec2Classify属于class 0的概率
    p0 = p0Vec[vec2Classify==1].sum() + np.log(1-pClass1)
    # 测试样本vec2Classify属于class 1的概率
    p1 = p1Vec[vec2Classify==1].sum() + np.log(pClass1)
    return int(p1 > p0)

def testingNB():
    # 读入原始数据
    listOPosts,listClasses = loadDataSet()
    # 生成vocabulary
    myVocabList = createVocabList(listOPosts)
    # 根据vovabulary把单词list转成vec
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 训练
    p0V,p1V,pAb=trainNB1(trainMat,listClasses)
    # 测试1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    # 测试2
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):
    # 将完整的样本按单词分割，且全部转成小写
    import re
    listOfTokens = re.split("\W+", bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 0]

def spamTest():
    # 读入原始数据，分割成单词的list
    dataSet = []
    for i in range(1,26):
        data = textParse(open('email/spam/'+str(i)+'.txt').read())
        dataSet.append(data)
    for i in range(1,26):
        data = textParse(open('email/ham/'+str(i)+'.txt',encoding='ISO-8859-15').read())
        dataSet.append(data)
    labels = np.hstack([np.zeros((26)), np.ones((26))])
    # 生成vocabulary
    myVocabList = createVocabList(dataSet)
    # 根据vovabulary把单词list转成vec
    dataVec = []
    for data in dataSet:
        dataVec.append(setOfWords2Vec(myVocabList, data))
    # 随机取40个数据为训练数据，10个数据为测试数据
    index = np.random.permutation(50)
    trainMat = np.array(dataVec)[index[10:]]
    trainLabel = np.array(labels)[index[10:]]
    # 训练
    p0V,p1V,pAb=trainNB1(trainMat,trainLabel)
    # 测试
    error = 0
    for i in index[:10]:
        label = classifyNB(dataVec[i],p0V,p1V,pAb)
        if label != labels[i]:
            error+=1
            print(dataSet[i])
    print("error rate = " + str(error/10))