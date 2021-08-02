def viterbi_algorithm(w1, transfer_features, w2, state_features, x, n_state):
    """维特比算法预测状态序列

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x: 需要计算的观测序列
    :param n_state: 状态的可能取值数
    :return: 最优可能的状态序列
    """
    n_transfer_features = len(transfer_features)  # 转移特征数
    n_state_features = len(state_features)  # 状态特征数
    n_position = len(x)  # 序列中的位置数

    # 定义状态矩阵
    dp = [[0.0] * n_state for _ in range(n_position)]  # 概率最大值
    last = [[-1] * n_state for _ in range(n_position)]  # 上一个结点

    # 处理t=0的情况
    for i in range(n_state):
        for l in range(n_state_features):
            dp[0][i] += w2[l] * state_features[l](y0=i, x=x, i=0)

    # 处理t>0的情况
    for t in range(1, n_position):
        for i in range(n_state):
            for j in range(n_state):
                d = dp[t - 1][i]
                for k in range(n_transfer_features):
                    d += w1[k] * transfer_features[k](y0=i, y1=j, x=x, i=t)
                for l in range(n_state_features):
                    d += w2[l] * state_features[l](y0=j, x=x, i=t)
                # print((i, j), "=", d)
                if d >= dp[t][j]:
                    dp[t][j] = d
                    last[t][j] = i
        # print(dp[t], last[t])

    # 计算最优路径的终点
    best_end, best_gamma = 0, 0
    for i in range(n_state):
        if dp[-1][i] > best_gamma:
            best_end, best_gamma = i, dp[-1][i]

    # 计算最优路径
    ans = [0] * (n_position - 1) + [best_end]
    for t in range(n_position - 1, 0, -1):
        ans[t - 1] = last[t][ans[t]]
    return ans


if __name__ == "__main__":
    import random

    def t1(y0, y1, x, i):
        return int(y0 == 0 and y1 == 1 and i in {1, 2})

    def t2(y0, y1, x, i):
        return int(y0 == 0 and y1 == 0 and i in {1})

    def t3(y0, y1, x, i):
        return int(y0 == 1 and y1 == 0 and i in {2})

    def t4(y0, y1, x, i):
        return int(y0 == 1 and y1 == 0 and i in {1})

    def t5(y0, y1, x, i):
        return int(y0 == 1 and y1 == 1 and i in {2})

    def s1(y0, x, i):
        return int(y0 == 0 and i in {0})

    def s2(y0, x, i):
        return int(y0 == 1 and i in {0, 1})

    def s3(y0, x, i):
        return int(y0 == 0 and i in {1, 2})

    def s4(y0, x, i):
        return int(y0 == 1 and i in {2})

    w1 = [1, 0.6, 1, 1, 0.2]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.5, 0.8, 0.5]
    s = [s1, s2, s3, s4]

    print(viterbi_algorithm(w1, t, w2, s, [random.randint(0, 1) for _ in range(3)], 2))  # [0, 1, 0]
