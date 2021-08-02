import random

import numpy as np


def k_means_clustering(X, k, random_state=0, max_iter=100):
    """k均值聚类算法

    :param X: 样本集
    :param k: 聚类数
    :param random_state: 随机种子
    :param max_iter: 最大迭代次数
    :return: 样本集合的聚类C
    """
    n_samples = len(X[0])

    # 随机选择k个样本点作为初始聚类中心
    random.seed(random_state)  # 选取随机种子
    means = [X[:, i] for i in random.sample(range(n_samples), k)]  # 将随机选择样本点作为初始聚类中心
    G0 = [[] for _ in range(k)]  # 每个初始聚类中心包含的样本点

    for _ in range(max_iter):
        G1 = [[] for _ in range(k)]

        # 对样本进行聚类
        for i in range(n_samples):
            c0, d0 = -1, float("inf")
            for c in range(k):
                d = np.sqrt((np.square(X[:, i] - means[c])).sum())
                if d < d0:
                    c0, d0 = c, d
            G1[c0].append(i)

        # 计算新的类中心
        change = False
        for c in range(k):
            mean = np.average([X[:, i] for i in G1[c]], axis=0)
            if not all(np.equal(mean, means[c])):
                means[c] = mean
                change = True

        if not change:
            break

        G0 = G1

    return G0


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])

    # 当随机种子(random_state)为1时，随机选择的初始聚类中心刚好与例14.2的解中的初始聚类中心相同
    print(k_means_clustering(X, 2, random_state=1))
