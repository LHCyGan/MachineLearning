import time
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

np.random.seed(32)

def preprocessing():
    # TODO:数据集查看和处理

    # 使用digits数据集进行聚类
    digits = load_digits()
    # 获取training-set
    data = digits.data
    # 获取label
    labels = digits.target
    # 查看相应维度
    num_insts, num_features = data.shape[0], data.shape[1]
    print("Number of instances in the dataset =", num_insts)
    print("Number of features =", num_features)

    # shuffle training-set
    # 生成同样instance 的 indexs
    indexs = np.arange(num_insts)
    print("Unshuffled indexs: ", indexs)
    # 打乱数据
    np.random.shuffle(indexs)
    print("Shuffled indexs: ", indexs)
    data = data[indexs, :]
    labels = labels[indexs]

    return num_insts, num_features, indexs, data, labels

def train():
    # TODO:搭建KMeans模型,并进行训练

    num_insts, num_features, indexs, data, labels = preprocessing()

    # 初始化模型参数
    num_clusters = 10
    kmeans_centers = np.zeros((num_clusters, num_features))
    # 随机选择10个instance作为clusters
    kmeans_centers = data[:num_clusters, :]
    print(kmeans_centers.shape)

    # TODO: Train
    num_iters = 10
    losses = np.zeros(num_iters)
    # 获取 start time
    start_time = time.process_time()
    for i in range(num_iters):
        # 计算每个 instance 到 cluster center 的 distance
        instance_square = np.sum(data * data, axis=1, keepdims=True)
        # print(instance_square.shape)
        center_square = np.sum(kmeans_centers * kmeans_centers, axis=1)
        # print(center_square.shape)
        # 计算内积
        inner_prod = np.dot(data, kmeans_centers.T)
        # print(inner_prod.shape)
        # 应用完全平方差公式
        distance = instance_square + center_square - 2 * inner_prod
        # compute loss
        losses[i] = np.mean(np.min(distance, axis=1))
        # 获取 new cluster
        new_cluster_idxs = np.argmin(distance, axis=1)
        # 更新 cluster
        for j in range(num_clusters):
            data_ = data[new_cluster_idxs == j, :]
            # 用 mean 更新
            kmeans_centers[j, :] = np.mean(data_, axis=0)
    # 获取 end time
    end_time = time.process_time()
    print("Time used for Kmeans clustering =", end_time - start_time, "seconds.")

    # 绘制损失函数
    plt.figure(figsize=(20, 8))
    plt.plot(np.arange(num_clusters), losses, 'o-', lw=3)
    plt.grid(alpha=0.5)
    plt.title("KMeans")
    plt.xlabel("Iteration Number")
    plt.ylabel("Objective Loss")
    plt.savefig("./kmeans.png")
    plt.show()


if __name__ == '__main__':
    train()