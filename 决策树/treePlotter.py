# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] + cntrPt[0]) / 2
    yMid = (parentPt[1] + cntrPt[1]) / 2
    createPlot.axl.text(xMid, yMid, txtString)


def plotNodeAndText(startPt, endPt, arrowTxt, nodeTxt, nodeType):
    plotNode(nodeTxt, endPt, startPt, nodeType)
    plotMidText(endPt, startPt, arrowTxt)


def leafId2x(leafId):
    # 根据leafID计算leaf的x坐标
    # 本例中，总宽度是1，leaf是3，所以每个leaf的x依次是1/6, 3/6, 5/6
    # leafId从1开始
    # leafId可以不是整数，例如1.5表示leaf1和leaf2中间的位置的x坐标
    return (leafId * 2 - 1) * plotTree.halfWidth


def rowId2y(rowId):
    # 根据rowId计算node/leaf的y坐标
    # root的rowId为0，root的子结点/叶子的rowID为1
    # 本例中，总高度是1，depth是2，所以每一层的高度是0.5，node的y依次是1， 0.5， 0
    return 1 - rowId * plotTree.height


def plotTree(myTree, parentPt, arrowTxt, rowId):
    feature = list(myTree.keys())[0]
    # cntrPT.x：一棵树的root结点的x坐标为其所有叶子的中间的点
    numLeafs = getNumLeafs(myTree)
    leafId = plotTree.finishedLeaf + (float(numLeafs) + 1) / 2
    nodePt = (leafId2x(leafId), rowId2y(rowId))
    plotNodeAndText(parentPt, nodePt, arrowTxt, feature, decisionNode)

    subTree = myTree[feature]
    parentPt = nodePt
    for key in subTree.keys():
        if type(subTree[key]).__name__ == 'dict':
            plotTree(subTree[key], parentPt, str(key), rowId + 1)
        else:
            plotTree.finishedLeaf += 1
            nodePt = (leafId2x(plotTree.finishedLeaf), rowId2y(rowId + 1))
            plotNodeAndText(parentPt, nodePt, str(key), subTree[key], leafNode)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    createPlot.leafs = getNumLeafs(inTree)
    plotTree.height = 1.0 / float(getTreeDepth(inTree))
    plotTree.halfWidth = 0.5 / float(getNumLeafs(inTree))
    plotTree.finishedLeaf = 0
    plotTree(inTree, (0.5, 1.0), '', rowId=0)  # root的parent就是root node上，所以第一点看不见箭头
    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': \
    {0: 'no', 1: 'yes'}}}},
    {'no surfacing': {0: 'no', 1: {'flippers': \
    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]

def getNumLeafs(myTree):
    numLeafs = 0
    feature = list(myTree.keys())[0]
    subTree = myTree[feature]
    for key in subTree:
        if type(subTree[key]).__name__=='dict':
            numLeafs += getNumLeafs(subTree[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    if type(myTree).__name__!='dict':
        return 1
    feature = list(myTree.keys())[0]
    subTree = myTree[feature]
    maxDepth = 0
    for key in subTree:
        if type(subTree[key]).__name__=='dict':
            depth = getTreeDepth(subTree[key])
        else:
            depth = 0
        if maxDepth < depth: maxDepth = depth
    return maxDepth+1
