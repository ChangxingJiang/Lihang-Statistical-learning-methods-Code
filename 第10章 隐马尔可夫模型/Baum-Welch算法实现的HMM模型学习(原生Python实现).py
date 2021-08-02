from random import random


def baum_welch(sequence, n_state, max_iter=100):
    """Baum-Welch算法学习隐马尔可夫模型

    :param sequence: 观测序列
    :param n_state: 可能的状态数
    :param max_iter: 最大迭代次数
    :return: A,B,π
    """
    n_samples = len(sequence)  # 样本数
    n_observation = len(set(sequence))  # 可能的观测数

    # ---------- 初始化随机模型参数 ----------
    # 初始化状态转移概率矩阵
    a = [[0.0] * n_state for _ in range(n_state)]
    for i in range(n_state):
        for j in range(n_state):
            a[i][j] = random()
        sum_ = sum(a[i])
        for j in range(n_state):
            a[i][j] /= sum_

    # 初始化观测概率矩阵
    b = [[0.0] * n_observation for _ in range(n_state)]
    for j in range(n_state):
        for k in range(n_observation):
            b[j][k] = random()
        sum_ = sum(b[j])
        for k in range(n_observation):
            a[j][k] /= sum_

    # 初始化初始状态概率向量
    p = [0.0] * n_state
    for i in range(n_state):
        p[i] = random()
    sum_ = sum(p)
    for i in range(n_state):
        p[i] /= sum_

    for _ in range(max_iter):
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

        # ---------- 计算：\gamma_t(i) ----------
        gamma = []
        for t in range(n_samples):
            sum_ = 0
            lst = [0.0] * n_state
            for i in range(n_state):
                lst[i] = alpha[t][i] * beta[t][i]
                sum_ += lst[i]
            for i in range(n_state):
                lst[i] /= sum_
            gamma.append(lst)

        # ---------- 计算：\xi_t(i,j) ----------
        xi = []
        for t in range(n_samples - 1):
            sum_ = 0
            lst = [[0.0] * n_state for _ in range(n_state)]
            for i in range(n_state):
                for j in range(n_state):
                    lst[i][j] = alpha[t][i] * a[i][j] * b[j][sequence[t + 1]] * beta[t + 1][j]
                    sum_ += lst[i][j]
            for i in range(n_state):
                for j in range(n_state):
                    lst[i][j] /= sum_
            xi.append(lst)

        # ---------- 计算新的状态转移概率矩阵 ----------
        new_a = [[0.0] * n_state for _ in range(n_state)]
        for i in range(n_state):
            for j in range(n_state):
                numerator, denominator = 0, 0
                for t in range(n_samples - 1):
                    numerator += xi[t][i][j]
                    denominator += gamma[t][i]
                new_a[i][j] = numerator / denominator

        # ---------- 计算新的观测概率矩阵 ----------
        new_b = [[0.0] * n_observation for _ in range(n_state)]
        for j in range(n_state):
            for k in range(n_observation):
                numerator, denominator = 0, 0
                for t in range(n_samples):
                    if sequence[t] == k:
                        numerator += gamma[t][j]
                    denominator += gamma[t][j]
                new_b[j][k] = numerator / denominator

        # ---------- 计算新的初始状态概率向量 ----------
        new_p = [1 / n_state] * n_state
        for i in range(n_state):
            new_p[i] = gamma[0][i]

        a, b, p = new_a, new_b, new_p

    return a, b, p


if __name__ == "__main__":
    # 根据例10.1的A,B,π生成的观测序列
    sequence = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
                0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]


    def forward_algorithm(a, b, p, sequence):
        """观测序列概率的前向算法"""
        n_state = len(a)  # 可能的状态数

        # 计算初值（定义状态矩阵）
        dp = [p[i] * b[i][sequence[0]] for i in range(n_state)]

        # 递推（状态转移）
        for k in range(1, len(sequence)):
            dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[k]] for i in range(n_state)]

        return sum(dp)


    A = [[0.0, 1.0, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.4, 0.0, 0.6], [0.0, 0.0, 0.5, 0.5]]
    B = [[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]]
    pi = [0.25, 0.25, 0.25, 0.25]
    print("生成序列的模型参数下，观测序列出现的概率:", forward_algorithm(A, B, pi, sequence))  # 6.103708248799872e-57

    A, B, pi = baum_welch(sequence, 4)
    print("训练结果的模型参数下，观测序列出现的概率:", forward_algorithm(A, B, pi, sequence))  # 8.423641064277491e-56
