import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

def preprocessing():
    # TODO:数据集查看和处理

    # 加载手写数据集
    mnist = fetch_mldata('MNIST original', data_home="./")
    data = mnist.data
    label = mnist.target
    num_instances, num_features = data.shape
    print("Number of data instances in MNIST =", num_instances)
    print("Number of feature per instance =", num_features)

    return data, label, num_instances, num_features

def train():
    # TODO:搭建PCA模型,并进行训练

    data, label, num_instances, num_features = preprocessing()

    # 1. step: compute the mean vector and remove the mean.
    data_mean = np.mean(data, axis=0)
    data_ = data - data_mean
    # 2. step: compute the covariance matrix.
    data_covariance = np.dot(data_.T, data_) / num_instances
    print("Size of the covariance matrix =", data_covariance.shape)
    # 3. step: compute the eigen-decomposition of the covariance matrix.
    eigvals, eigvectors = np.linalg.eigvals(data_covariance)  # default:ascending but need descending
    indexs = np.arange(num_instances - 1, -1, -1)
    eigvals = eigvals[indexs]
    eigvectors = eigvectors[indexs]
    # 4. step: Select projection dimension. Compute the projection and its reconstruction image
    k = 50
    project = eigvectors[:, k]
    # Compute the projection image.
    project_img = np.dot(data, project)
    # Compute the reconstruction image
    reconstruction_img = np.dot(project_img, project.T)
    # 5. step: visualize the original image and the reconstructed image
    for j in range(10):
        idx = np.random.randint(num_instances)
        fig = plt.figure()
        plt.gray()
        fig.add_subplot(1, 2, 1)
        plt.imshow(data[idx, :].reshape(28, 28))
        fig.add_subplot(1, 2, 2)
        plt.imshow(reconstruction_img[idx, :].reshape(28, 28))
