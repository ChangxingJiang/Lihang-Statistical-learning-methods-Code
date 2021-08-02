import numpy as np


def get_stationary_distribution(P, tol=1e-8, max_iter=1000):
    """迭代法求离散有限状态马尔可夫链的某个平稳分布

    根据平稳分布的定义求平稳分布。如果有无穷多个平稳分布，则返回其中任意一个。如果不存在平稳分布，则无法收敛。

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 平稳分布
    """
    n_components = len(P)

    # 初始状态分布：均匀分布
    pi0 = np.array([1 / n_components] * n_components)

    # 迭代寻找平稳状态
    for _ in range(max_iter):
        pi1 = np.dot(P, pi0)

        # 判断迭代更新量是否小于容差
        if np.sum(np.abs(pi0 - pi1)) < tol:
            break

        pi0 = pi1

    return pi0


def is_reversible(P, tol=1e-4, max_iter=1000):
    """计算有限状态马尔可夫链是否可逆

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 可逆 = True ; 不可逆 = False
    """
    n_components = len(P)
    D = get_stationary_distribution(P, pow(tol, 2), max_iter)  # 计算平稳分布
    for i in range(n_components):
        for j in range(n_components):
            if not - tol < P[i][j] * D[j] - P[j][i] * D[i] < tol:
                return False
    return True


if __name__ == "__main__":
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(is_reversible(P))  # True

    P = np.array([[0.25, 0.5, 0.25],
                  [0.25, 0, 0.5],
                  [0.5, 0.5, 0.25]])

    print(is_reversible(P))  # False
