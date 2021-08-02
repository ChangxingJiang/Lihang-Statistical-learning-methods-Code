import numpy as np


def correlation_coefficient(X):
    """计算所有样本之间的相关系数矩阵"""
    n_samples = len(X[0])

    # 计算均值
    means = np.mean(X, axis=0)

    # 计算误差平方和
    variance = [np.square((X[:, i] - means[i])).sum() for i in range(n_samples)]

    # 构造相关系数矩阵
    D = np.identity(n_samples)  # 初始化为单位矩阵

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = X[:, i], X[:, j]
            numerator = ((xi - means[i]) * (xj - means[j])).sum()
            denominator = np.sqrt(variance[i] * variance[j])
            if denominator:
                D[i][j] = D[j][i] = numerator / denominator
            else:  # 当出现零方差时
                D[i][j] = D[j][i] = np.nan

    return D


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])

    print(correlation_coefficient(X))

    # [[ 1. nan -1. -1. -1.]
    #  [nan  1. nan nan nan]
    #  [-1. nan  1.  1.  1.]
    #  [-1. nan  1.  1.  1.]
    #  [-1. nan  1.  1.  1.]]
