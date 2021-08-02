import numpy as np


def pagerank_2(M, d=0.8, tol=1e-8, max_iter=1000):
    """计算一般PageRank的幂法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
    """
    n_components = len(M)

    # 选择初始向量x0：均匀分布
    x0 = np.array([1 / n_components] * n_components)

    # 计算有向图的一般转移矩阵A
    A = d * M + (1 - d) / n_components

    # 迭代并规范化结果向量
    for _ in range(max_iter):
        x1 = np.dot(A, x0)
        x1 /= np.max(x1)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(x0 - x1)) < tol:
            break

        x0 = x1

    # 对结果进行规范化处理，使其表示概率分布
    x0 /= np.sum(x0)

    return x0


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_2(P))  # [0.2  0.27 0.27 0.27]

    P = np.array([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 1, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0]])

    print(pagerank_2(P))  # [0.1  0.13 0.64 0.13]

    P = np.array([[0, 0, 1],
                  [1 / 2, 0, 0],
                  [1 / 2, 1, 0]])

    print(pagerank_2(P))  # [0.38 0.22 0.4 ]
