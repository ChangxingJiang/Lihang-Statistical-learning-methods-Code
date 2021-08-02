import numpy as np


def em_for_plsa(X, K, max_iter=100, random_state=0):
    """概率潜在语义模型参数估计的EM算法

    :param X: 单词-文本共现矩阵
    :param K: 话题数量
    :param max_iter: 最大迭代次数
    :param random_state: 随机种子
    :return: P(w_i|z_k)和P(z_k|d_j)
    """
    n_features, n_samples = X.shape

    # 计算n(d_j)
    N = [np.sum(X[:, j]) for j in range(n_samples)]

    # 设置参数P(w_i|z_k)和P(z_k|d_j)的初始值
    np.random.seed(random_state)
    P1 = np.random.random((n_features, K))  # P(w_i|z_k)
    P2 = np.random.random((K, n_samples))  # P(z_k|d_j)

    for _ in range(max_iter):
        # E步
        P = np.zeros((n_features, n_samples, K))
        for i in range(n_features):
            for j in range(n_samples):
                for k in range(K):
                    P[i][j][k] = P1[i][k] * P2[k][j]
                P[i][j] /= np.sum(P[i][j])

        # M步
        for k in range(K):
            for i in range(n_features):
                P1[i][k] = np.sum([X[i][j] * P[i][j][k] for j in range(n_samples)])
            P1[:, k] /= np.sum(P1[:, k])

        for k in range(K):
            for j in range(n_samples):
                P2[k][j] = np.sum([X[i][j] * P[i][j][k] for i in range(n_features)]) / N[j]

    return P1, P2


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 2, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0]])

    np.set_printoptions(precision=2, suppress=True)
    R1, R2 = em_for_plsa(X, 3)

    print(R1)
    # [[0.   0.15 0.  ]
    #  [0.15 0.   0.  ]
    #  [0.   0.   0.4 ]
    #  [0.15 0.   0.  ]
    #  [0.08 0.08 0.  ]
    #  [0.23 0.31 0.4 ]
    #  [0.   0.15 0.  ]
    #  [0.15 0.   0.  ]
    #  [0.23 0.   0.  ]
    #  [0.   0.15 0.2 ]
    #  [0.   0.15 0.  ]]

    print(R2)
    # [[0. 0. 0. 0. 0. 1. 1. 0. 1.]
    #  [1. 0. 1. 1. 1. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 1. 0.]]
