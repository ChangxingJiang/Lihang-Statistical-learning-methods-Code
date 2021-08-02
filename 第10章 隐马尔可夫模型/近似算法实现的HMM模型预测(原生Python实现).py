def approximation_algorithm(a, b, p, sequence):
    """近似算法预测状态序列"""
    n_samples = len(sequence)
    n_state = len(a)  # 可能的状态数

    # ---------- 计算：前向概率 ----------
    # 计算初值（定义状态矩阵）
    dp = [p[i] * b[i][sequence[0]] for i in range(n_state)]
    alpha = [dp]

    # 递推（状态转移）
    for t in range(1, n_samples):
        dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[t]] for i in range(n_state)]
        alpha.append(dp)

    # ---------- 计算：后向概率 ----------
    # 计算初值（定义状态矩阵）
    dp = [1] * n_state
    beta = [dp]

    # 递推（状态转移）
    for t in range(n_samples - 1, 0, -1):
        dp = [sum(a[i][j] * dp[j] * b[j][sequence[t]] for j in range(n_state)) for i in range(n_state)]
        beta.append(dp)

    beta.reverse()

    # 计算最优可能的状态序列
    ans = []
    for t in range(n_samples):
        min_state, min_gamma = -1, 0
        for i in range(n_state):
            gamma = alpha[t][i] * beta[t][i]
            if gamma > min_gamma:
                min_state, min_gamma = i, min_gamma
        ans.append(min_state)
    return ans


if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(approximation_algorithm(A, B, p, [0, 1, 0]))  # [2, 2, 2]
