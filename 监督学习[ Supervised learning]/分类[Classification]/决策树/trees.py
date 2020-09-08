# -*- encoding:utf-8 -*-
# author: liuheng
from math import log
import operator

class DecisionTree:
    def __init__(self):
        self.dataset = self.createDataset()

    def createDataset(self):
        """创建数据集"""
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        # change to discrete values
        return dataSet, labels

    def calcShannonEnt(self):
        """计算信息熵"""
        # 计算数据集中实例的总数。
        numEntries = len(self.dataset)
        # 创建一个数据字典，它的键值是最后一列的数值
        labelCounts = {}
        for featVec in self.dataset:
            currentLabel = featVec[-1]
            # 每个键值都记录了当前类别出现的次数
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            # 使用所有类标签的发生频率计算类别出现的概率
            prob = float(labelCounts[key]) / numEntries
            # 将用这个概率计算香农熵
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def splitDataset(self, axis, value):
        """
        :param axis: 划分数据集的特征
        :param value: 需要返回的特征的值
        """
        # 创建一个新的列表对象
        retDataset = []
        for featVec in self.dataset:
            # 按照某个特征划分数据集时，就需要将所有符合要求的元素抽取出来
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataset.append(reducedFeatVec)
        return retDataset

    def chooseBestFeatureToSplit(self):
        numFeatures = len(self.dataset[0]) - 1
        baseEntropy = self.calcShannonEnt()
        bestInfoGain = 0.0; bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in self.dataset]
            uniqueVals = set(featList)  # 创建唯一的分类标签列表
            newEntropy = 0.0
            # 计算每种划分方式的信息熵
            for value in uniqueVals:
                subDataSet = self.splitDataset(i, value)
                prob = len(subDataSet) / float(len(self.dataset))
                self.dataset = subDataSet
                newEntropy += prob * self.calcShannonEnt()
            infoGain = baseEntropy - newEntropy
            # 计算最好的信息增益
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def majorityCnt(self, classList: "分类名称的列表"):
        # 创建键值为classList中唯一值的数据字典
        classCount = {}
        # 字典对象存储了classList中每个类标签出现的频率，最后利用operator操作键值排序字典，并返回出现次数最多的分类名称。
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, labels):
        # 先创建了名为classList的列表变量，其中包含了数据集的所有类标签
        classList = [example[-1] for example in self.dataset]
        # 递归函数的第一个停止条件是所有的类标签完全相同，则直接返回该类标签
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 递归函数的第二个停止条件是使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
        if len(self.dataset) == 1:
            # 由于第二个条件无法简单地返回唯一的类标签，这里使用majorityCnt函数挑选出现次数最多的类别作为返回值。
            return self.majorityCnt(classList)
        # 当前数据集选取的最好特征存储在变量bestFeat中，得到列表包含的所有属性值
        bestFeat = self.chooseBestFeatureToSplit(self.dataset)
        bestFeatLabel = labels[bestFeat]
        # 字典变量myTree存储了树的所有信息
        myTree = {bestFeatLabel:{}}
        del labels[bestFeat]
        """
        遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数
        createTree()，得到的返回值将被插入到字典变量myTree中，因此函数终止执行时，字典中将
        会嵌套很多代表叶子节点信息的字典数据
        """
        featValues = [example[bestFeat] for example in self.dataset]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            # 复制类标签
            subLabels = labels[:]
            """
            因为在Python语言中函数参数是列表类型时，参数是按照引用方式传递的。为了保
            证每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表。
            """
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataset(
                bestFeat, value
            ), subLabels)
        return myTree

    def __call__(self, labels):
        return self.createTree(labels)