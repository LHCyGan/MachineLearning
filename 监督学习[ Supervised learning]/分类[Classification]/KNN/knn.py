# -*- encoding:utf-8 -*-
import numpy as np
import operator

def createDataset():
    """使用createDataSet()函数，它创建数据集和标签"""
    group = np.array([[1.0, 1.1],
                      [1.0, 1.0],
                      [0, 0],
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def KNN(X: "用于分类的输入向量",
        dataSet: "输入的训练样本集", labels: "标签向量",
        k: "示用于选择最近邻居的数目"):
    """搭建knn model"""
    """
    对未知类别属性的数据集中的每个点依次执行以下操作：
    (1) 计算已知类别数据集中的点与当前点之间的距离；
    (2) 按照距离递增次序排序；
    (3) 选取与当前点距离最小的k个点；
    (4) 确定前k个点所在类别的出现频率；
    (5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
    """
    # 使用欧氏距离公式，计算两个向量点xA和xB之间的距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(X, (dataSetSize, 1)) - dataSet
    sqDiffMat = np.square(diffMat)
    sqDistances = np.sum(sqDiffMat, axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndices = distances.argsort()
    classCount = {}
    # 确定前k个距离最小元素所在的主要分类
    for i in range(k):
        votedLabel = labels[sortedDistIndices[i]]
        classCount[votedLabel] = classCount.get(votedLabel, 0) + 1
    # 导入运算符模块的itemgetter方法，按照第二个元素的次序对元组进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


