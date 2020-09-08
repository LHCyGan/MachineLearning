import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def sigmoid(x):
    """sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """softmax 函数"""
    # x.shape == (n_features, m)
    x_e = np.exp(x)
    x_sum = np.sum(x_e, axis=0, keepdims=True)
    s = x_e / x_sum
    return s

def relu(x):
    """relu 函数"""
    return np.maximum(x, 0)


def compute_loss(a3, Y):
    """
    Implement the loss function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    loss - value of the loss function
    """

    m = Y.shape[1]
    # 对数似然函数 / 交叉熵损失函数
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    # 求均值
    loss = 1. / m * np.nansum(logprobs)  # 将np.nan当作0来计算

    return loss


def forward_propagation(X, parameters):
    """前向传播"""

    # 获取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Linear
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    # 记录参数，用于backward propagation
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache


def backward_propagation(X, Y :"Label", cache: "参数tuple"):
    """反向传播"""

    m = X.shape[1]
    z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3 = cache

    dz3 = 1. / m * (a3 - Y)  # 对数似然loss
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients

def update_parameters(parameters, grads, learning_rate):
    """参数应用梯度更新"""

    L = len(parameters) // 2  # 获取网络的层数

    # 更新参数
    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters


def predict(X, y, parameters):

    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    # 前向传播
    a3, cache = forward_propagation(X, parameters)

    # predict
    for i in range(0, m):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

    return p


def load_dataset(is_plot=True):
    # 制造数据集
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # 可视化
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def plot_decision_boundary(model, X, y):
    """ 绘制决策边界 """
    # 设置界限
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[0, :].max() + 1
    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    # 获取网格的函数值
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # 绘制
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X, threshold=0.5):
    """ 预测边界 """
    a3, cache = forward_propagation(X, parameters)
    predictions = a3 > 0.5
    return predictions
