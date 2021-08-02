from bisect import bisect_left
from copy import deepcopy
from random import random


def build_markov_sequence(a, b, p, t):
    """隐马尔可夫模型观测序列的生成"""

    # 计算前缀和
    a = deepcopy(a)
    for i in range(len(a)):
        for j in range(1, len(a[0])):
            a[i][j] += a[i][j - 1]
    b = deepcopy(b)
    for i in range(len(b)):
        for j in range(1, len(b[0])):
            b[i][j] += b[i][j - 1]
    p = deepcopy(p)
    for j in range(1, len(p)):
        p[j] += p[j - 1]

    # 按照初始状态分布p产生状态i
    stat = [bisect_left(p, random())]

    # 按照状态的观测概率分布生成观测o
    res = [bisect_left(b[stat[-1]], random())]

    for _ in range(1, t):
        # 按照状态的状态转移概率分布产生下一个状态
        stat.append(bisect_left(a[stat[-1]], random()))

        # 按照状态的观测概率分布生成观测o
        res.append(bisect_left(b[stat[-1]], random()))

    return res


if __name__ == "__main__":
    A = [[0.0, 1.0, 0.0, 0.0],
         [0.4, 0.0, 0.6, 0.0],
         [0.0, 0.4, 0.0, 0.6],
         [0.0, 0.0, 0.5, 0.5]]
    B = [[0.5, 0.5],
         [0.3, 0.7],
         [0.6, 0.4],
         [0.8, 0.2]]
    p = [0.25, 0.25, 0.25, 0.25]
    print(build_markov_sequence(A, B, p, 200))
