# -*- encoding:utf-8 -*-
# author: liuheng
from collections import Counter, defaultdict
import numpy as np
import operator


class NBayes(object):
    def __init__(self, smooth=1):
        self.smooth = smooth  # 贝叶斯估计方法的平滑参数smooth=1，当smooth=0时即为最大似然估计
        self.p_prior = {}  # 先验概率
        self.p_condition = {}  # 条件概率

    def train(self, vector_data, label_data):

        n_samples = label_data.shape[0]  # 计算样本数
        # 统计不同类别的样本数，并存入字典，key为类别，value为样本数
        # Counter类的目的是用来跟踪值出现的次数。以字典的键值对形式存储，其中元素作为key，其计数作为value。
        dict_label = Counter(label_data)
        K = len(dict_label)
        for key, val in dict_label.items():
            # 计算先验概率
            self.p_prior[key] = (val + self.smooth) / (n_samples + K * self.smooth)

        # 计算后验概率
        # 分别对每个特征维度进行计算，vector_data.shape[1]为特征向量的维度
        for d in range(vector_data.shape[1]):
            # defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
            nums_vd = defaultdict(int)
            # 抽取特定维度
            vector_d = vector_data[:, d]
            # unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
            nums_s = len(np.unique(vector_d))  # 每个特征向量的可能取值个数
            for xd, y in zip(vector_d, label_data):
                nums_vd[(xd, y)] += 1
            for key, val in nums_vd.items():
                #  d为维度，key[0]为每个特征向量每个维度的的值, key[1]为类别
                self.p_condition[(d, key[0], key[1])] = (val + self.smooth) / (
                            dict_label[key[1]] + nums_s * self.smooth)

    # 预测未知特征向量的类别
    def predict(self, input_v):
        p_predict = {}
        # y为类别，p_y为每个类别的先验概率
        for y, p_y in self.p_prior.items():
            p = p_y  # 计算每种后验概率
            for d, v in enumerate(input_v):
                p *= self.p_condition[(d, v, y)]
            p_predict[y] = p
        #     对字典按value进行排序
        p_predict_sorted = sorted(p_predict.items(), key=operator.itemgetter(1), reverse=True)
        # 获取字典中value最大值所对应键的大小
        # return max(p_predict, key=p_predict.get)
        return p_predict_sorted[0]


if __name__ == "__main__":
    # 以《统计学习方法》中的例4.1计算，为方便计算，将例子中"S"设为0，“M"设为1。
    data = np.array([[1, 0, -1], [1, 1, -1], [1, 1, 1], [1, 0, 1],
                     [1, 0, -1], [2, 0, -1], [2, 1, -1], [2, 1, 1],
                     [2, 2, 1], [2, 2, 1], [3, 2, 1], [3, 1, 1],
                     [3, 1, 1], [3, 2, 1], [3, 2, -1]])
    # 提取特征向量
    vector_data = data[:, :-1]
    # 提取label类别
    label_data = data[:, -1]
    # 采用贝叶斯估计计算条件概率和先验概率，此时拉普拉斯平滑参数为1，为0时即为最大似然估计
    bayes = NBayes(smooth=1)
    # 实例化
    bayes.train(vector_data, label_data)
    # 朴素贝叶斯分类
    print(bayes.predict(np.array([2, 0])))
