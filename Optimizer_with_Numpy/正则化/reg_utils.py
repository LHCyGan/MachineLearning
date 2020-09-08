import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def initialize_parameters(layers_dim):
    """初始化权重参数"""
    np.random.seed(3)
    parameters = {}
    L = len(layers_dim)
    # Xavier
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) / np.sqrt(layers_dim[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))

        assert parameters['W' + str(l)].shape == (layers_dim[l], layers_dim[l-1])
        assert parameters['b' + str(l)].shape == (layers_dim[l], 1)

    return parameters

def forward_propagation(X, parameters):
    """构建前向传播过程"""
    # 获取参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # 构建线性层
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    # 保存参数
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache


def compute_cost(a3, Y):
    m = Y.shape[1]
    logprobs = - (Y * np.log(a3) + (1 - Y) * np.log(1 - a3))
    cost = (1. / m) * np.nansum(logprobs)

    return cost


def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    # 计算梯度
    dz3 = 1. / m * (a3 - Y)
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


def update_parameters(parameters, grads, learning_rate=0.1):
    L = len(parameters) // 2

    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters

def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    # forward propagation
    a3, cache = forward_propagation(X, parameters)
    # a3.shape == (1, m)
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

    return p

def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def plot_desicion_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    # Predict the function value for the whole grid
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions

