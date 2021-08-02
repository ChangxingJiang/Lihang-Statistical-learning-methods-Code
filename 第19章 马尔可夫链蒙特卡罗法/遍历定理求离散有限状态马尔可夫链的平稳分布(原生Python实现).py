from bisect import bisect_left

import numpy as np


def get_stationary_distribution(P, start_iter=1000, end_iter=2000, random_state=0):
    """遍历定理求离散有限状态马尔可夫链的某个平稳分布

    要求离散状态、有限状态马尔可夫链是不可约、非周期的。

    :param P: 转移概率矩阵
    :param start_iter: 认为多少次迭代之后状态分布就是平稳分布
    :param end_iter: 计算从start_iter次迭代到end_iter次迭代的状态分布
    :param random_state: 随机种子
    :return: 平稳分布
    """
    n_components = len(P)
    np.random.seed(random_state)

    # 计算累计概率用于随机抽样
    Q = P.T
    for i in range(n_components):
        for j in range(1, n_components):
            Q[i][j] += Q[i][j - 1]

    # 设置初始状态
    x = 0

    # start_iter次迭代
    for _ in range(start_iter):
        v = np.random.rand()
        x = bisect_left(Q[x], v)

    F = np.zeros(n_components)
    # start_iter次迭代到end_iter次迭代
    for _ in range(start_iter, end_iter):
        v = np.random.rand()
        x = bisect_left(Q[x], v)
        F[x] += 1

    return F / sum(F)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(get_stationary_distribution(P))  # [0.39 0.18 0.43]

    P = np.array([[1, 1 / 3, 0],
                  [0, 1 / 3, 0],
                  [0, 1 / 3, 1]])

    print(get_stationary_distribution(P))  # [1. 0. 0.]
