import numpy as np


def scatter_matrix(X, G):
    """计算样本散布矩阵

    :param X: 样本
    :param G: 类别包含的样本
    :return: 样本散步矩阵
    """
    n_samples = len(G)
    n_features = len(X)

    # 计算类的中心
    means = np.mean(X[:, G], axis=1)

    A = np.zeros((n_features, n_features))
    for i in range(n_samples):
        A += np.dot((X[:, i] - means)[:, np.newaxis], (X[:, i] - means)[:, np.newaxis].T)

    return A


def covariance_matrix(X, G):
    """计算样本协方差矩阵"""
    n_features = len(X)
    A = scatter_matrix(X, G)
    S = A / (n_features - 1)
    return S


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])

    print(scatter_matrix(X, [2, 3, 4]))
    # [[34.         -0.66666667]
    #  [-0.66666667  2.66666667]]

    print(covariance_matrix(X, [2, 3, 4]))
    # [[34.         -0.66666667]
    #  [-0.66666667  2.66666667]]
