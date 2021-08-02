import numpy as np


def mahalanobis_distance(X):
    """计算所有样本之间的马哈拉诺比斯距离矩阵"""
    n_samples = len(X[0])

    # 计算协方差矩阵
    S = np.cov(X)

    # 计算协方差矩阵的逆矩阵
    S = np.linalg.inv(S)

    # 构造马哈拉诺比斯距离矩阵
    D = np.zeros((n_samples, n_samples))  # 初始化为零矩阵

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi = X[:, i][:, np.newaxis]
            xj = X[:, j][:, np.newaxis]
            D[i][j] = D[j][i] = np.sqrt((np.dot(np.dot((xi - xj).T, S), (xi - xj))))

    return D


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])

    print(mahalanobis_distance(X))

    # [[0.         1.83604716 1.91649575 2.81058198 1.94257172]
    #  [1.83604716 0.         0.38851434 1.94257172 2.52783249]
    #  [1.91649575 0.38851434 0.         1.55405738 2.27648631]
    #  [2.81058198 1.94257172 1.55405738 0.         1.83604716]
    #  [1.94257172 2.52783249 2.27648631 1.83604716 0.        ]]
