import numpy as np

def loadDataSet(fileName):
    dataSet = []
    labels = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataSet.append(dataList[0:-1])
        labels.append(dataList[-1])
    return dataSet, labels

def generateI(dataMat, labelMat, alphas, i, b, toler, C):
    # 如果这一轮要更新alpha_i，先把alpha_i对应的样本信息都准备好
    I = generateNode(dataMat, labelMat, alphas, i, b)
    # condition to choose i: error is big enough
    if(not choose(I, toler, C)):
        raise UserWarning('error not big enough')
    return I

def generateNode(dataMat, labelMat, alphas, index, b):
    return {'index':index, # 更新第几个alpha
             'x':dataMat[index,:], # 对应的数据
             'y':labelMat[index],     # 对应的标签
             'a':alphas[index],       # 当前的alpha值
            #E: error between predition class and real class
            # 基于当前alpha对第index个数据的预测分类与该数据的真实分类做比较
             'E':E(dataMat, labelMat, alphas, index, b)
           }

def E(dataMat, labelMat, alphas, i, b):
    # fx: the predition of the class
    fXi = float(np.multiply(alphas, labelMat).T*(dataMat*dataMat[i,:].T)) + b
    # E: error between predition class and real class
    Ei = fXi - labelMat[i]
    return Ei

# If this error is large, then the alpha corresponding to this data instance can be optimized.
# ???
def choose(I, toler, C):
    return ((I['y']*I['E'] < - toler) and (I['a'] < C)) or ((I['y']*I['E'] > toler) and (I['a'] > 0))

def generateJ(dataMat, labelMat, alphas, i, b, m):
    # select j randomly
    j = selectJrand(i, m)
    return generateNode(dataMat, labelMat, alphas, j, b)

# i: the index of first alpha
# m: the total number of alphas
# choose a random valuse which is not equal to i
def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(np.random.uniform(0, m))
    return j

# 更新alpha_i, alpha_j和b，使得目标函数进一步变大
def update(I, J, C, b):
    # 根据公式先更新alpha_j
    update_alpha_j(I, J, C)
    # change alpha_i as alpha_j changed
    update_alpha_i(I, J)
    # 当更新了一对a_i,a_j之后，需要重新计算b。
    b = update_b(I, J, C, b)
    return I['a'], J['a'], b

def product(I, J):
    return I['x'] * J['x'].T

def calc_eta(I, J):
    eta = 2.0 * product(I, J) - product(I, I) - product(J, J)
    if eta >= 0:
        raise UserWarning('eta>=0')
    return eta

def update_alpha_j(I, J, C):
    # eta: the optional amount to change alpha[j]
    eta = calc_eta(I, J)
    alphas_j = J['a']-J['y'] *(I['E'] - J['E']) / eta
    # make sure alpha_j is in [0, C]
    L, H = calcLH(C, I, J)   # 此处要用到更新前的J['a']和I['a']
    alphas_j = clipAlpha(alphas_j, H, L)
    J['gap'] = alphas_j - J['a']
    J['a'] = alphas_j
    if(abs(J['gap']) < 0.00001):# the float way to compare
        raise UserWarning('J not moving enough')

def calcLH(C, I, J):
    if (I['y'] != J['y']):
        L = max(0, J['a'] - I['a'])
        H = min(C, C+J['a']-I['a'])
    else:
        L = max(0, I['a'] + J['a'] - C)
        H = min(C, I['a'] + J['a'])
    if L == H:
        raise UserWarning('L==H')
    return L, H

# 将aj限制在H和L之间
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def update_alpha_i(I, J):
    I['gap'] = I['y']*J['y']*(-J['gap'])
    I['a'] = I['a'] + I['gap']

def update_b(I, J, C, b):
    b1 = b + calc_b_gap(I, J, I)
    b2 = b + calc_b_gap(I, J, J)
    if (not isInBound(I, C)):
        b = b1
    elif (not isInBound(J, C)):
        b = b2
    else:
        b = (b1 + b2)/2.0
    return b

def isInBound(I, C):
    return not (0 < I['a']) and (I['a'] < C)

def calc_b_gap(I, J, target):
    return float(-target['E'] - I['y']*product(I, target)*(I['gap']) - J['y']*product(J, target)*(J['gap']))

# 使用SMO算法求出alpha，然后根据alpha计算w和b
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat, labelMat = np.mat(dataMatIn), np.mat(classLabels).transpose()
    m= np.shape(dataMat)[0]
    alphas = np.mat(np.zeros((m, 1)))# 把所有alpha都初始化为0
    b, iter = 0, 0
    # 迭代matIter次
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            try: # 可能会有一些情况导致选择的这一对ahpha不能更新
                # 选择一个要更新的alpha
                I = generateI(dataMat, labelMat, alphas, i, b, toler, C)
                # 选择另一个要更新的alpha
                J = generateJ(dataMat, labelMat, alphas, i, b, m)
                # 更新alpha_i, alpha_j，b也需要相应的更新
                alphas[i], alphas[J['index']], b = update(I, J, C, b)
                alphaPairsChanged += 1
                print ("iter: %d, i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            except UserWarning as err:
                print (err)
        if alphaPairsChanged == 0:
            iter+=1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

# 根据alphas计算w
# ???
def get_w(alphas, dataset, labels):
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T * np.array([1,1]) *dataset
    w = np.dot(yx.T, alphas)
    return w.T[0]

import matplotlib.pyplot as plt

def showSimpleSMO(dataArr, labelArr, alphas, b):
    dataMat, labelMat = np.array(dataArr), np.array(labelArr)

    # 绘制样本点
    plt.scatter(dataMat[labelMat==1,0], dataMat[labelMat==1,1])
    plt.scatter(dataMat[labelMat==-1,0], dataMat[labelMat==-1,1])

    # 绘制决策边界
    w = get_w(alphas, dataMat, labelMat)
    print (w, b)
    x = np.array([0, 10])
    y = (-b - x*w[0])/w[1]
    plt.plot(x, y)

    # 绘制支撑向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x,y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='#AB3319')
    plt.show()

# --------------------------------------------------------------------------------------------------------------

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kernel=product):
        self.X = np.mat(dataMatIn)
        self.labelMat = np.mat(classLabels).transpose()
        self.C = C
        self.toler = toler
        self.m= np.shape(self.X)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))# 把所有alpha都初始化为0
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) # m*2, [i,0]代表[i,1]值是否有效。[i,1]代表E_i
        self.kernel = kernel
        #self.K = kernel(self.X)

# X: m*n
# y: 1*n向量映射成1*m的向量
def kernelTrans(X, y, kernel):
    if kernel == product:
        temp = y * X.T
    else:
        k = np.mat(np.zeros((X.shape[0], 1)))
        for j in range(X.shape[0]):
            k[j] = (X[j] - y)*(X[j] - y).T
        temp = np.exp(-k/(smoP.sigma**2)).T
    return temp

# 相当于上面的def E(dataMat, labelMat, alphas, i, b)
def calcEk(oS, i):
    # fx: the predition of the class
    temp = kernelTrans(oS.X, oS.X[i], oS.kernel)
    fXi = float(temp * np.multiply(oS.alphas, oS.labelMat)) + oS.b
    # E: error between predition class and real class
    Ei = fXi - oS.labelMat[i]
    return Ei

# 类似于前面的def selectJrand(i, m):
def selectJ(i, oS, Ei):
    maxj,maxEj,maxDeltaE = 0, 0, -1
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache.A[:, 0])
    if len(validEcacheList) > 1: # len至少为1，因为至少eCache[i]=1。但如果len=1，说明这是第一次运行，只有eCache[i]是有效的。
        for j in validEcacheList:
            Ej = calcEk(oS, j)
            # choose the second alpha so that we’ll take the maximum step during each optimization
            if abs(Ej-Ei) > maxEj:  #
                maxk, maxEj, maxDeltaE = j, Ej, abs(Ej-Ei)
    else:
        maxj = selectJrand(i, oS.m)
        maxEj = calcEk(oS, maxj)
    return maxj, maxEj

def updateEk(oS, k):
    oS.eCache[k] = [1, calcEk(oS, k)]


def choose_2(oS, I):
    return ((I['y'] * I['E'] < - oS.toler) and (I['a'] < oS.C)) or ((I['y'] * I['E'] > oS.toler) and (I['a'] > 0))


def generateNode_2(oS, index):
    return {'index': index,  # 更新第几个alpha
            'x': oS.X[index, :],  # 对应的数据
            'y': oS.labelMat[index],  # 对应的标签
            'a': oS.alphas[index],  # 当前的alpha值
            # E: error between predition class and real class
            # 基于当前alpha对第index个数据的预测分类与该数据的真实分类做比较
            'E': calcEk(oS, index)
            }


def generateI_2(oS, i):
    # 如果这一轮要更新alpha_i，先把alpha_i对应的样本信息都准备好
    I = generateNode_2(oS, i)
    # condition to choose i: error is big enough
    if (not choose_2(oS, I)):
        raise UserWarning('error not big enough')
    return I


def generateJ_2(oS, I):
    # select j randomly
    j, _ = selectJ(I['index'], oS, I['E'])
    return generateNode_2(oS, j)


def updateEk_2(oS, K):
    k = K['index']
    oS.alphas[k] = K['a']
    oS.eCache[k] = [1, calcEk(oS, k)]

def gaussianKernel(I, J):
    return - (J['x'] - I['x']).dot((J['x'] - I['x']).T) / (smoP.sigma**2)

def calc_eta_2(K, I, J):
    eta = 2.0 * K(I, J) - K(I, I) - K(J, J)
    if eta >= 0:
        raise UserWarning('eta>=0')
    return eta

def update_alpha_j_2(oS, I, J):
    # eta: the optional amount to change alpha[j]
    eta = calc_eta_2(oS.kernel, I, J)
    alphas_j = J['a']-J['y'] *(I['E'] - J['E']) / eta
    # make sure alpha_j is in [0, C]
    L, H = calcLH(oS.C, I, J)   # 此处要用到更新前的J['a']和I['a']
    alphas_j = clipAlpha(alphas_j, H, L)
    J['gap'] = alphas_j - J['a']
    J['a'] = alphas_j
    if(abs(J['gap']) < 0.00001):# the float way to compare
        raise UserWarning('J not moving enough')

def update_b_2(oS, I, J):
    b1 = oS.b + calc_b_gap_2(I, J, I, oS.kernel)
    b2 = oS.b + calc_b_gap_2(I, J, J, oS.kernel)
    if (not isInBound(I, oS.C)):
        b = b1
    elif (not isInBound(J, oS.C)):
        b = b2
    else:
        b = (b1 + b2)/2.0
    return b

def calc_b_gap_2(I, J, target, K):
    return float(-target['E'] - I['y']*K(I, target)*(I['gap']) - J['y']*K(J, target)*(J['gap']))

# 更新alpha_i, alpha_j和b，使得目标函数进一步变大
def update_2(oS, I, J):
    # 根据公式先更新alpha_j
    update_alpha_j_2(oS, I, J)
    updateEk_2(oS, J)
    # change alpha_i as alpha_j changed
    update_alpha_i(I, J)
    updateEk_2(oS, I)
    # 当更新了一对a_i,a_j之后，需要重新计算b。
    oS.b = update_b_2(oS, I, J)
    # return I['a'], J['a'], oS.b


def innerL(i, oS):
    try:
        I = generateI_2(oS, i)
        J = generateJ_2(oS, I)
        update_2(oS, I, J)
    except UserWarning as err:
        # 因为打印太多，把它屏蔽掉了
        # print(err)  # 打印报错的字符串
        return 0
    return 1


def smoP(dataMatIn, classLabels, C, toler, maxIter, kernel=product, sigma =0.1):
    smoP.sigma = sigma
    oS = optStruct(dataMatIn, classLabels, C, toler, kernel)
    iter = 0
    entireSet, alphaPairsChanged = True, 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet == True)):
        alphaPairsChanged = 0
        iter += 1
        l = np.arange(oS.m)
        if not entireSet:
            l = l[np.array(oS.alphas.A[:, 0] > 0) & np.array(oS.alphas.A[:, 0] < C)]  # oS.alphas.A[:,0]为什么是list不是array?
        for i in l:
            alphaPairsChanged += innerL(i, oS)
        print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if entireSet == True:
            entireSet = False
        elif alphaPairsChanged == 0:
            if entireSet == False:
                entireSet = True
    return oS


def testWithFile(filename, supportVectors, labelSV, alphas, svInd, b):
    dataArr, labelArr = loadDataSet(filename)
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        # 把测试数据集从向量空间升级到kernel向量空间
        # 这个转换过程不需要全部的数据集，只要支撑向量就可以了
        # datMat[i,:] -> kernelEval
        kernelEval = kernelTrans(supportVectors, datMat[i, :], gaussianKernel)  # 1*m
        # 把新的空间上的向量代入公式wx+b预测y
        predict = kernelEval.dot((np.multiply(labelSV, alphas[svInd])).T) + b
        # 不是判断相等，而是判断正负号，因为正负代表分类，具体的数值的绝对值代表分类的可信度
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

def testRbf(sigma=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    oS = smoP(dataArr, labelArr, 200, 0.0001, 1000, gaussianKernel, sigma)
    b, alphas = oS.b, oS.alphas.T.A[0]
    svInd = alphas > 1e-3
    datMat = np.array(dataArr);
    labelMat = np.array(labelArr).transpose()
    supportVectors = datMat[alphas > 1e-3]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % supportVectors.shape[0])
    return  supportVectors, labelSV, alphas, svInd, b

def img2vector(filename):
    ret = np.zeros((0))
    fr = open(filename)
    for line in fr.readlines():
        line = line[:32]
        newinfo = np.array(list(line), dtype=int)
        ret = np.hstack([ret, newinfo])
    return ret.reshape(1,-1)

from os import listdir
def loadImages(dirName):
    labels = []
    dataSet = np.zeros((0,1024))
    trainingFileList = listdir('digits/'+dirName)
    for file in trainingFileList:
        digit = int(file.split('_')[0])
        if digit == 9:
            labels.append(-1)
        else:
            labels.append(1)
        dataSet = np.vstack([dataSet, img2vector('digits/'+dirName+'/'+file)])
    return dataSet, labels

def testDigits(kernel, sigma):
    dataArr,labelArr = loadImages('trainingDigits')
    oS = smoP(dataArr, labelArr, 200, 0.0001, 10000, kernel, sigma)
    b, alphas = oS.b, oS.alphas.T.A[0]
    svInd = alphas > 1e-3
    datMat = np.array(dataArr);
    labelMat = np.array(labelArr).transpose()
    supportVectors = datMat[alphas > 1e-3]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % supportVectors.shape[0])
    return  supportVectors, labelSV, alphas, svInd, b

def testDigitWithFile(filename, supportVectors, labelSV, alphas, svInd, b):
    dataArr, labelArr = loadImages(filename)
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        # 把测试数据集从向量空间升级到kernel向量空间
        # 这个转换过程不需要全部的数据集，只要支撑向量就可以了
        # datMat[i,:] -> kernelEval
        kernelEval = kernelTrans(supportVectors, datMat[i, :], gaussianKernel)  # 1*m
        # 把新的空间上的向量代入公式wx+b预测y
        predict = kernelEval.dot((np.multiply(labelSV, alphas[svInd])).T) + b
        # 不是判断相等，而是判断正负号，因为正负代表分类，具体的数值的绝对值代表分类的可信度
        # print(predict, labelArr[i])
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
