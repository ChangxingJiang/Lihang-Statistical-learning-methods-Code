import numpy as np


def is_reducible(P):
    """计算马尔可夫链是否可约

    :param P: 转移概率矩阵
    :return: 可约 = True ; 不可约 = False
    """
    n_components = len(P)

    # 遍历所有状态k，检查从状态k出发能否到达任意状态
    for k in range(n_components):
        visited = set()  # 当前已遍历过的状态

        find = False  # 当前是否已找到可到达任意位置的时刻
        stat0 = (False,) * k + (True,) + (False,) * (n_components - k - 1)  # 时刻0可达到的位置

        while stat0 not in visited:
            visited.add(stat0)
            stat1 = [False] * n_components

            for j in range(n_components):
                if stat0[j] is True:
                    for i in range(n_components):
                        if P[i][j] > 0:
                            stat1[i] = True

            # 如果已经到达之前已检查可到达任意状态的状态，则不再继续寻找
            for i in range(k):
                if stat1[i] is True:
                    find = True
                    break

            # 如果当前时刻可到达任意位置，则不再寻找
            if all(stat1) is True:
                find = True
                break

            stat0 = tuple(stat1)

        if not find:
            return True

    return False


if __name__ == "__main__":
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(is_reducible(P))  # False

    P = np.array([[0, 0.5, 0],
                  [1, 0, 0],
                  [0, 0.5, 1]])

    print(is_reducible(P))  # True
