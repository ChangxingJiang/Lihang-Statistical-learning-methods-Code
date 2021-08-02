import numpy as np


def pagerank_basic(M, tol=1e-8, max_iter=1000):
    """使用PageRank的基本定义求解PageRank值

    要求有向图是强联通且非周期性的

    :param M: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 初始状态分布：均匀分布
    pr0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pr1 = np.dot(M, pr0)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pr0 - pr1)) < tol:
            break

        pr0 = pr1

    return pr0


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    P = np.array([[0, 1 / 2, 1, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_basic(P))  # [0.33 0.22 0.22 0.22]

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_basic(P))  # [0. 0. 0. 0.]
