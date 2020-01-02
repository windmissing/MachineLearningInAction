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

def K(I, J):
    return I['x'] * J['x'].T

def calc_eta(I, J):
    eta = 2.0 * K(I, J) - K(I, I) - K(J, J)
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
    print(b1, b2)
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
    return float(-target['E'] - I['y']*K(I, target)*(I['gap']) - J['y']*K(J, target)*(J['gap']))

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