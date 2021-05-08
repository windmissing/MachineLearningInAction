# -*- coding: utf-8 -*-
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None  # link similar items (the dashed lines in figure 12.1)
        self.parent = parentNode
        self.children = {}  # an empty dictionary for the children of this node

    #  increments the  count variable by a given amount
    def inc(self, numOccur):
        self.count += numOccur

    # display the tree in text, for debugging
    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)  # 左对齐显示
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    itemCount, headerTable = {}, {}
    # The first pass goes through everything in the dataset and counts the frequency of each term.
    for transaction in dataSet:
        for item in transaction:
            itemCount[item] = itemCount.get(item, 0) + 1  # dataSet[trans]
    print('item count = ', itemCount)
    # the header table is scanned and items occurring less than  minSup are deleted.
    for key in itemCount.keys():
        if itemCount[key] >= minSup:
            headerTable[key] = [itemCount[key],
                                None]  # he header table hold a count and pointer to the first item of each type.
    print('headerTable = ', headerTable)
    # create the base node, which contains the null set Ø
    rootNode = treeNode('Null Set', 1, None)
    # iterate over the dataset again
    for transaction in dataSet:
        # 把frequent低的item过滤掉
        localD = []
        for item in transaction:
            if item in headerTable:  # 在headerTable中的都是Frequent达到要求的
                localD.append(item)
        # 过滤之后排序
        localD.sort(key=lambda k: (headerTable[k][0]), reverse=True)
        # print ('localD = ', localD)
        # 将得到的新的Transaction插入到tree中
        node = rootNode

        for item in localD:
            node = updateTree(item, node, headerTable, 1)
        # print ('tree after one transaction', rootNode.disp())
    return rootNode, headerTable


# 把item插入到以node为父结点的树中
def updateTree(item, node, headerTable, count):
    if item in node.children:
        node.children[item].count += count
    else:
        node.children[item] = treeNode(item, count, node)
        if headerTable[item][1] == None:
            headerTable[item][1] = node.children[item]
        else:
            updateHeader(headerTable[item][1], node.children[item])
    return node.children[item]


# headerTable和后面结点的nodeLink组成一个单链表
def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
    ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
    ['z'],
    ['r', 'x', 'n', 'o', 's'],
    ['y', 'r', 'x', 'z', 'q', 't', 'p'],
    ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def ascendTree(treeNode):  # 这个node不一定是leaf
    prefix = []
    while(treeNode.parent != None):
        prefix.append(treeNode.name)
        treeNode = treeNode.parent
    return prefix

# basePath: 暂时没用上
# treeNode: headTable[item]
def findPrefixPath(treeNode):
    prefixPaths = {}
    while(treeNode != None):
        prefix = ascendTree(treeNode)
        if len(prefix) > 1:
            prefixPaths[frozenset(prefix[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return prefixPaths


# freqItemList：所有满足minSup的Set都存到这里
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # sorting the items in the header table by their frequency of occurrence
    sortedKey = [item[0] for item in sorted(headerTable.items(), key=lambda p: p[0])]
    # Construct cond. FP-tree from cond. pattern base
    for key in sortedKey:
        # preFix和{preFix,key}都是满足minSup的set，应该都存在于freqItemList的中，因为要copy
        # 如果不copy，freqItemList中只有{preFix,key}，而preFix就丢失了
        newFreqSet = preFix.copy()
        newFreqSet.add(key)
        # each frequent item is added to your list of frequent itemsets
        freqItemList.append(newFreqSet)

        # 新的一轮迭代
        # create a conditional base
        condPattBases = findPrefixPath(headerTable[key][1])
        # print(key, condPattBases)
        # This conditional base is treated as a new dataset and fed to  createTree()
        myCondTree, myHead = createTree_2(condPattBases, minSup)
        # Mine cond. FP-tree
        if len(myHead) > 0:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

# createTree假设所有Data出现即frequency+1
# 在现在的应该场景中，dataSet的每一项是个dict，dict的Value说明dict的key中的data出现一次代表frequency+几
def createTree_2(dataSet, minSup=1):
    itemCount, headerTable = {}, {}
    # The first pass goes through everything in the dataset and counts the frequency of each term.
    for transaction in dataSet.keys():
        for item in transaction:
            itemCount[item] = itemCount.get(item, 0) + dataSet[transaction]
    #print ('item count = ', itemCount)
    # the header table is scanned and items occurring less than  minSup are deleted.
    for key in itemCount.keys():
        if itemCount[key] >= minSup:
            headerTable[key] = [itemCount[key], None] # he header table hold a count and pointer to the first item of each type.
    #print ('headerTable = ',headerTable)
    # create the base node, which contains the null set Ø
    rootNode = treeNode('Null Set', 1, None)
    # iterate over the dataset again
    for transaction in dataSet:
        # 把frequent低的item过滤掉
        localD = []
        for item in transaction:
            if item in headerTable: # 在headerTable中的都是Frequent达到要求的
                localD.append(item)
        # 过滤之后排序
        localD.sort(key=lambda k:(headerTable[k][0]), reverse=True)
        #print ('localD = ', localD)
        # 将得到的新的Transaction插入到tree中
        node = rootNode
        for item in localD:
            node = updateTree(item, node, headerTable, dataSet[transaction])
        #print ('tree after one transaction', rootNode.disp())
    return rootNode, headerTable

