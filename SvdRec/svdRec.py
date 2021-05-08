import numpy as np
from numpy import linalg as la

# 欧式距离函数 -> 相似度
def ecludSim(inA,inB):
    # la.form求A和B的欧式距离
    # 距离与相似度程相反的关系，因此取倒数
    # 为防止分母为0，求得的距离再加1
    # 计算结果为(0, 1]
    # 缺点：一个特征不一样就会导致距离很大
    #print(inA, inB)
    return 1.0/(1.0 + la.norm(inA - inB))

# pearson相似度
def pearsSim(inA,inB):
    # 判断两组数据与某一直线的拟合程度
    # 优点：数据不规范时效果好
    if len(inA) < 3 : return 1.0
    # 结果为[-1,1]，需要转到[0,1]区间
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

# 余弦相似度：只考虑两组数据之间的夹角
def cosSim(inA,inB):
    # 不会因为文章的长度不同导致结果偏差太大
    # 结果为[-1,1]，需要转到[0,1]区间
    num = float(inA.T.dot(inB))
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def loadExData():
    return[[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def standEst(dataMat, user, simMeas, dishId):
    totalSim, totalScore = 0, 0
    # 找出user对其它dish的点评
    for j in range(dataMat.shape[1]):
        if dataMat[user, j] == 0: continue
        # 假如user对j有点评，找出对dishID和j都有点评的人
        overlap = np.array(dataMat[:, j] > 0) & np.array(dataMat[:, dishId] > 0)
        # 根据这些人对j的评价和对dishId的评价，计算j和dishId的相似度
        if dataMat[overlap].shape[0] == 0:
            sim = 0
        else:
            sim = simMeas(dataMat[overlap, j], dataMat[overlap, dishId])
        # 以相似度为权值，根据user对j的评价来估计user对dishId的评价
        totalScore += sim * dataMat[user, j]
        totalSim += sim
    if totalScore == 0: return 0
    return totalScore / totalSim


# dataMat:一行代表一个User，一列代表一个菜
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 1 Look for things the user hasn’t yet rated: look for values with 0 in the user-item matrix.
    notRatedDishId = np.arange(dataMat.shape[1])[dataMat[user, :] == 0]
    # 2 Of all the items this user hasn’t yet rated, find a projected rating for each item:
    # that is, what score do we think the user will give to this item?
    notRatedScorePredict = []
    for dishId in notRatedDishId:
        # print (dishId)
        notRatedScorePredict.append((dishId, estMethod(dataMat, user, simMeas, dishId)))
    # 3 Sort the list in descending order and return the first N items.
    notRatedScorePredict.sort(key=lambda p: p[1], reverse=True)
    return notRatedScorePredict

def loadExData_2():
    return[[1, 1, 0, 2, 2],
            [2, 0, 0, 3, 3],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def svdEst(dataMat, user, simMeas, dishId):
    # 先对dataMat做SVD分解
    U,Sigma,VT = la.svd(dataMat)
    # 把Sigma转成对角矩阵
    Sig4 = np.eye(4) * Sigma[:4]
    # 对dataMat变形，仅包含前4个奇异特征
    # 不知道书上为什么是U*sigma*data.T，但前面的介绍说是U*sigma*VT
    # 计算结果与书上不太一样，但总体上差不多
    xformedItems = U[:,:4].dot(Sig4).dot(VT[:4,:])
    #xformedItems = (dataMat.T.dot(U[:,:4]).dot(np.mat(Sig4).I)).T
    totalSim, totalScore = 0, 0
    # 找出user对其它dish的点评
    for j in range(dataMat.shape[1]):
        if dataMat[user, j] == 0:continue
        # 为什么要把overlap那一段去掉了呢？
        sim = simMeas(xformedItems[:, j], xformedItems[:, dishId]) # 计算相似度时使用去噪之后的data
        print ('the %d and %d similarity is: %f' % (dishId, j, sim))
        # 以相似度为权值，根据user对j的评价来估计user对dishId的评价
        totalScore += sim * dataMat[user, j]
        totalSim += sim
    if totalScore == 0:return 0
    return totalScore/totalSim

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print ('1,',end="")
            else: print ('0,',end="")
        print ('')

def imgCompress(numSV=3, thresh=0.8):
    ret = np.zeros((0,32))
    fr = open('0_5.txt')
    for line in fr.readlines():
        line = line[:32]
        newinfo = np.array(list(line), dtype=int)
        ret = np.vstack([ret, newinfo])
    myMat = ret.copy()#reshape(1,-1)
    print (myMat)
    print ("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigN = np.eye(numSV) * Sigma[:numSV]
    #SigN = np.eye(numSV)
    #for k in range(numSV):
    #    SigN[k,k] = Sigma[k]
    reconMat = U[:,:numSV].dot(SigN).dot(VT[:numSV,:])
    print ("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)