import numpy as np


def pca_by_svd(X, k):
    """数据矩阵奇异值分解进行的主成分分析算法

    :param X: 样本矩阵X
    :param k: 主成分个数k
    :return:
    """
    n_samples = X.shape[1]

    # 构造新的n×m矩阵
    T = X.T / np.sqrt(n_samples - 1)

    # 对矩阵T进行截断奇异值分解
    U, S, V = np.linalg.svd(T)
    V = V[:, :k]

    # 求k×n的样本主成分矩阵
    return np.dot(V.T, X)


if __name__ == "__main__":
    X = np.array([[2, 3, 3, 4, 5, 7],
                  [2, 4, 5, 5, 6, 8]])
    X = X.astype("float64")

    # 规范化变量
    avg = np.average(X, axis=1)
    var = np.var(X, axis=1)
    for i in range(X.shape[0]):
        X[i] = (X[i, :] - avg[i]) / np.sqrt(var[i])

    print(pca_by_svd(X, 2))

    # [[-2.02792041 -0.82031104 -0.4330127   0.          0.82031104  2.46093311]
    #  [ 0.2958696  -0.04571437 -0.4330127   0.          0.04571437  0.1371431 ]]
