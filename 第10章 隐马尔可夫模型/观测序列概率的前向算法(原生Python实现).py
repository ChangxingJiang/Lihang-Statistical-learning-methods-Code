def forward_algorithm(a, b, p, sequence):
    """观测序列概率的前向算法"""
    n_state = len(a)  # 可能的状态数

    # 计算初值（定义状态矩阵）
    dp = [p[i] * b[i][sequence[0]] for i in range(n_state)]

    # 递推（状态转移）
    for k in range(1, len(sequence)):
        dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[k]] for i in range(n_state)]

    return sum(dp)


if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(forward_algorithm(A, B, p, [0, 1, 0]))  # 0.130218
