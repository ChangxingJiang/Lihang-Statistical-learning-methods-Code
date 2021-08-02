from collections import Counter

import numpy as np


def is_periodic(P):
    """计算马尔可夫链是否有周期性

    :param P: 转移概率矩阵
    :return: 有周期性 = True ; 无周期性 = False
    """
    n_components = len(P)

    # 0步转移概率矩阵
    P0 = P.copy()
    hash_P = tuple(P0.flat)

    # 每一个状态上一次返回状态的时刻的最大公因数
    gcd = [0] * n_components

    visited = Counter()  # 已遍历过的t步转移概率矩阵
    t = 1  # 当前时刻t

    # 不断遍历时刻t，直至满足如下条件：当前t步转移矩阵之前已出现过2次（至少2次完整的循环）
    while visited[hash_P] < 2:
        visited[hash_P] += 1

        # 记录当前返回状态的状态
        for i in range(n_components):
            if P0[i][i] > 0:
                if gcd[i] == 0:  # 状态i出发时，还从未返回过状态i
                    gcd[i] = t
                else:  # 计算最大公约数
                    gcd[i] = np.gcd(gcd[i], t)

        # 检查当前时刻是否还有未返回(gcd[i]=0)或返回状态的所有时间长的最大公因数大于1(gcd[i]>1)的状态
        for i in range(n_components):
            if gcd[i] == 0 or gcd[i] > 1:
                break
        else:
            return False

        # 计算(t+1)步转移概率矩阵
        P1 = np.dot(P0, P)

        P0 = P1
        hash_P = tuple(P0.flat)
        t += 1

    return True


if __name__ == "__main__":
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])

    print(is_periodic(P))  # False

    P = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])

    print(is_periodic(P))  # True
