def loadDataSet(fileName):
    dataSet = []
    labels = []
    for line in open(fileName).readlines():
        dataList = [float(data) for data in line.strip().split('\t')]
        dataSet.append(dataList[0:-1])
        labels.append(dataList[-1])
    return dataSet, labels