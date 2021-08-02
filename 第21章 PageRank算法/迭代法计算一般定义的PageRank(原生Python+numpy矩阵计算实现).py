import numpy as np


def pagerank_1(M, d=0.8, tol=1e-8, max_iter=1000):
    """PageRank的迭代算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 初始状态分布：均匀分布
    pr0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pr1 = d * np.dot(M, pr0) + (1 - d) / n_components

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pr0 - pr1)) < tol:
            break

        pr0 = pr1

    return pr0


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_1(P))  # [0.1  0.13 0.13 0.13]

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 1, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_1(P))  # [0.1  0.13 0.64 0.13]

    P = np.array([[0, 0, 1],
                  [1 / 2, 0, 0],
                  [1 / 2, 1, 0]])

    print(pagerank_1(P))  # [0.38 0.22 0.4 ]
