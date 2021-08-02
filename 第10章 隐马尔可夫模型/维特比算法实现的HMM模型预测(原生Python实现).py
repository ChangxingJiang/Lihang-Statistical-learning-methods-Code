def viterbi_algorithm(a, b, p, sequence):
    """维特比算法预测状态序列"""
    n_samples = len(sequence)
    n_state = len(a)  # 可能的状态数

    # 定义状态矩阵
    dp = [[0.0] * n_state for _ in range(n_samples)]  # 概率最大值
    last = [[-1] * n_state for _ in range(n_samples)]  # 上一个结点

    # 处理t=0的情况
    for i in range(n_state):
        dp[0][i] = p[i] * b[i][sequence[0]]

    # 处理t>0的情况
    for t in range(1, n_samples):
        for i in range(n_state):
            for j in range(n_state):
                delta = dp[t - 1][j] * a[j][i]
                if delta >= dp[t][i]:
                    dp[t][i] = delta
                    last[t][i] = j
            dp[t][i] *= b[i][sequence[t]]

    # 计算最优路径的终点
    best_end, best_gamma = 0, 0
    for i in range(n_state):
        if dp[-1][i] > best_gamma:
            best_end, best_gamma = i, dp[-1][i]

    # 计算最优路径
    ans = [0] * (n_samples - 1) + [best_end]
    for t in range(n_samples - 1, 0, -1):
        ans[t - 1] = last[t][ans[t]]
    return ans


if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(viterbi_algorithm(A, B, p, [0, 1, 0]))  # [2, 2, 2]
