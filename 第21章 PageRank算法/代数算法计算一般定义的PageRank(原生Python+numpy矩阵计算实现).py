import numpy as np


def pagerank_3(M, d=0.8, tol=1e-8, max_iter=1000):
    """PageRank的代数算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 计算第一项：(I-dM)^-1
    r1 = np.linalg.inv(np.diag([1] * n_components) - d * M)

    # 计算第二项：(1-d)/n 1
    r2 = np.array([(1 - d) / n_components] * n_components)

    return np.dot(r1, r2)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_3(P))  # [0.1  0.13 0.13 0.13]

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 1, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_3(P))  # [0.1  0.13 0.64 0.13]

    P = np.array([[0, 0, 1],
                  [1 / 2, 0, 0],
                  [1 / 2, 1, 0]])

    print(pagerank_3(P))  # [0.38 0.22 0.4 ]
