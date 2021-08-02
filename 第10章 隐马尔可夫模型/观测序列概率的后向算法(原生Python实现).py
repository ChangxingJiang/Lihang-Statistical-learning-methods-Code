def backward_algorithm(a, b, p, sequence):
    """观测序列概率的后向算法"""
    n_state = len(a)  # 可能的状态数

    # 计算初值（定义状态矩阵）
    dp = [1] * n_state

    # 递推（状态转移）
    for k in range(len(sequence) - 1, 0, -1):
        dp = [sum(a[i][j] * dp[j] * b[j][sequence[k]] for j in range(n_state)) for i in range(n_state)]

    return sum(p[i] * b[i][sequence[0]] * dp[i] for i in range(n_state))


if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(backward_algorithm(A, B, p, [0, 1, 0]))  # 0.130218
