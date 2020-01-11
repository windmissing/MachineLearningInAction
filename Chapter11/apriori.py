import numpy as np

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createCandidate(dataSet):
    candidate = set()
    for data in dataSet:
        for item in data:
            candidate.add(frozenset([item])) # dict的key需要可以hash的值。普通的set不是可以hash的值。forzenset才可以。frozen是不可变的set。
    return list(candidate)

def scanD(transaction, candidate1, minSupport):
    values = {}
    for can in candidate1:  # 防止有些can从来没有出现在transaction中过
        values[can] = 0
    # For each transaction in tran the dataset:
    for trans in transaction:
        # For each candidate itemset, can:
        for can in candidate1:
            # Check to see if can is a subset of tran
            if can.issubset(trans):
                # If so increment the count of can
                values[can] = values.get(can, 0) + 1
    # For each candidate itemset:
    items = []
    supportValue = {}
    for can in candidate1:
        # If the support meets the minimum, keep this item
        if values[can]/len(transaction) >= minSupport:
            supportValue[can] = values[can]/len(transaction)
            items.append(can)
    return items, values

def getTransaction(dataSet):
    D = []
    for data in dataSet:
        D.append(set(data))
    return D


# Ck:一个Frozen包含k个元素，过滤support之前的列表
# Lk:一个frozen包含k个元素，过滤support之后的列表
# C1: createCandidate
# Ck->Lk: scanD
# Lk -> Ck+1：generateCk
def generateCk(Lk, k):
    retList = []
    lenLK = len(Lk)
    for i in range(lenLK):  # 遍历包含LK，
        for j in range(i + 1, lenLK):
            L1 = list(Lk[i])[:k - 2];
            L2 = list(Lk[j])[:k - 2]
            # L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    # C1 = apriori.createCandidate(dataSet)
    transaction = getTransaction(dataSet)
    # L1,suppData0=apriori.scanD(transaction, C1, 0.5)
    # While the number of items in the set is greater than 0:
    k = 1
    L = []
    support = {}
    while True:
        if k == 1:
            Ck = createCandidate(dataSet)
        else:
            Ck = generateCk(Lk, k)
        # print ('Ck=',Ck)
        Lk, supportk = scanD(transaction, Ck, minSupport)
        support.update(supportk)
        # print ('Lk=',Lk)
        k += 1
        L.append(Lk)
        if len(Lk) == 0:
            break

        # Create a list of candidate itemsets of length k
        # Ck = GenerateCk(Lk, K)
        # Scan the dataset to see if each itemset is frequent
        # Keep frequent itemsets to create itemsets of length k+1
    return L, support

# 计算出来的Lk的list
# supportData: 计算出来的每一个Lk的frequency
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for k in range (1, len(L)): # 从L2开始，遍历整个L，每一项是一个list
        for frozen in L[k]: # L[k]是第K+1层L，遍历这一层的L，每一项是一个forzenset
            HSet = [frozenset([item]) for item in frozen]
            if k == 1: # 只计算两个单个的item之间的关系(frozen代表PH，里面有2个元素，HSet代表H,里面有一个元素)
                rulesFromConseq(frozen, HSet, supportData, bigRuleList, minConf)
            else: # (frozen代表PH，里面有3个及以上的元素，HSet代表H,里面有一个元素，此时HSet中的元素可以组合成新的Case)
                calculateConfidence(frozen, HSet, supportData, bigRuleList, minConf)
    return bigRuleList

def calculateConfidence(PHSet, HSetList, supportData, bigRuleList, minConf=0.7):
    newHSetList = generateCk(HSetList, len(HSetList[0]) + 1) # 一个元素的HSet升级成2个元素的newHSet
    while(len(PHSet) > len(newHSetList[0])): # PHSet比newHSet中的元素多
        rulesFromConseq(PHSet, newHSetList, supportData, bigRuleList, minConf)
        newHSetList = generateCk(newHSetList, len(newHSetList[0]) + 1) # 尝试继续升级

# PHSet：L[k-1]中的每一项，是个forzenset
# HSetList：forzenset组成的list，即rule(P->H)中的H的list
# supportData: 计算出来的每一个Lk的frequency
def rulesFromConseq(PHSet, HSetList, supportData, bigRuleList, minConf=0.7):
    #print (PHSet, HSetList, supportData)
    for HSet in HSetList:
        # P|H 是 PHSet。H是HSetList里的一个item。P是P|H-H，PHSet-HSet
        # confidence(p-->H)= frequency(P|H)/frequency(P)
        confidence = supportData[PHSet] / supportData[PHSet - HSet]
        # print ((PHSet-HSet), '-->', HSet, supportData[PHSet], supportData[PHSet - HSet], confidence)
        if confidence > minConf:
            print ((PHSet-HSet), '-->', HSet, confidence)
            bigRuleList.append(((PHSet-HSet), HSet, confidence))