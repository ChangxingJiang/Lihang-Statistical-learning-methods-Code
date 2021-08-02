import bisect
import math
import random

from scipy.misc import derivative


def newton_method_linear(func, args=(), error=1e-6, dx=1e-6):
    """一维牛顿法求f(x)=0的值

    :param func: 目标函数
    :param args: 参数列表
    :param error: 容差
    :param dx: 计算导数时使用的dx
    :return:
    """
    x0, y0 = 0, func(0, *args)
    while True:
        d = derivative(func, x0, args=args, dx=dx)  # 计算一阶导数
        x1 = x0 - y0 / d
        if abs(x1 - x0) < error:
            return x1
        x0, y0 = x1, func(x1, *args)


def count_conditional_probability(w1, t, w2, s, x, y):
    """已知条件随机场模型计算状态序列关于观测序列的非规范化条件概率

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x: 需要计算的观测序列
    :param y: 需要计算的状态序列
    :return: 状态序列关于观测序列的条件概率
    """
    n_features_1 = len(w1)  # 转移特征数
    n_features_2 = len(w2)  # 状态特征数
    n_position = len(x)  # 序列中的位置数

    res = 0
    for k in range(n_features_1):
        for i in range(1, n_position):
            res += w1[k] * t[k](y[i - 1], y[i], x, i)
    for k in range(n_features_2):
        for i in range(n_position):
            res += w2[k] * s[k](y[i], x, i)
    return pow(math.e, res)


def make_hidden_sequence(w1, t, w2, s, x_range, y_range, n_samples=1000, random_state=0):
    """已知模型构造随机样本集

    :param w1: 模型的转移特征权重
    :param t: 模型的转移特征函数
    :param w2: 模型的状态特征权重
    :param s: 模型的状态特征函数
    :param x_range: 观测序列的可能取值
    :param y_range: 状态序列的可能取值
    :param n_samples: 生成样本集样本数(近似)
    :return: 状态序列关于观测序列的条件概率
    """
    P = [[0.0] * len(y_range) for _ in range(len(x_range))]  # 条件概率分布

    lst = []
    sum_ = 0
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            P[i][j] = round(count_conditional_probability(w1, t, w2, s, x, y), 1)
            sum_ += P[i][j]
            lst.append(sum_)

    X, Y = [], []

    random.seed(random_state)
    for _ in range(n_samples):
        r = random.uniform(0, sum_)
        idx = bisect.bisect_left(lst, r)
        i, j = divmod(idx, len(y_range))
        X.append(x_range[i])
        Y.append(y_range[j])

    return X, Y


def improved_iterative_scaling(x, y, transfer_features, state_features, tol=1e-4, max_iter=1000):
    """改进的迭代尺度法学习条件随机场模型

    :param x: 输入变量
    :param y: 输出变量
    :param transfer_features: 转移特征函数
    :param state_features: 状态特征函数
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 条件随机场模型
    """
    n_samples = len(x)  # 样本数
    n_transfer_features = len(transfer_features)  # 转移特征数
    n_state_features = len(state_features)  # 状态特征数

    # 坐标压缩
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 观测序列可能取值数
    n_y = len(y_list)  # 状态序列可能取值数
    n_total = n_x * n_y  # 观测序列可能取值和状态序列可能取值的笛卡尔积

    print(x_list, x_mapping)
    print(y_list, y_mapping)

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    # print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    # print("边缘分布的经验分布", d2)

    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features

    # 计算转移特征函数关于经验分布的期望值:EP(tk) (empirical_joint_distribution_each_feature)
    d3 = [0.0] * n_transfer_features  # empirical_joint_distribution_each_feature
    for k in range(n_transfer_features):
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                for i in range(1, n_position):
                    if transfer_features[k](y[i - 1], y[i], x, i):
                        d3[k] += d1[xi][yi]
                        nn[xi][yi] += 1

    # 计算状态特征函数关于经验分布的期望值:EP(sl) (empirical_joint_distribution_each_feature)
    d4 = [0.0] * n_state_features  # empirical_joint_distribution_each_feature
    for l in range(n_state_features):
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                for i in range(n_position):
                    if state_features[l](y[i], x, i):
                        d4[l] += d1[xi][yi]
                        nn[xi][yi] += 1

    # print("转移特征函数关于经验分布的期望值:", d3)
    # print("状态特征函数关于经验分布的期望值:", d4)
    # print("所有特征在(x,y)出现的次数:", nn)

    # 定义w的初值和模型P(Y|X)的初值
    w1 = [0] * n_transfer_features  # w的初值：wi=0
    w2 = [0] * n_state_features  # w的初值：wi=0
    p0 = [[1 / n_total] * n_y for _ in range(n_x)]  # 当wi=0时，P(Y|X)的值

    for _ in range(max_iter):
        change = False

        # 遍历所有转移特征以更新w
        for k in range(n_transfer_features):
            def func(d, kk):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        xx = x_list[xxi]
                        yy = y_list[yyi]
                        n_position = len(x_list[xxi])  # 序列中的位置数
                        val = 0
                        for i in range(1, n_position):
                            val += transfer_features[kk](yy[i - 1], yy[i], xx, i)
                        val *= d2[xxi] * p0[xxi][yyi] * pow(math.e, d * nn[xxi][yyi])
                        res += val
                res -= d3[kk]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(k,))

            # 更新wi的值
            w1[k] += dj
            if abs(dj) >= tol:
                change = True

        for l in range(n_state_features):
            def func(d, ll):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        xx = x_list[xxi]
                        yy = y_list[yyi]
                        n_position = len(x_list[xxi])  # 序列中的位置数
                        val = 0
                        for i in range(n_position):
                            val += state_features[ll](yy[i], xx, i)
                        val *= d2[xxi] * p0[xxi][yyi] * pow(math.e, d * nn[xxi][yyi])
                        res += val
                res -= d4[ll]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(l,))

            # 更新wi的值
            w2[l] += dj
            if abs(dj) >= tol:
                change = True

        # 计算新的模型
        p1 = [[0.0] * n_y for _ in range(n_x)]
        for xi in range(n_x):
            for yi in range(n_y):
                x = x_list[xi]
                y = y_list[yi]
                n_position = len(x_list[xi])  # 序列中的位置数
                res = 0
                for k in range(n_transfer_features):
                    for i in range(1, n_position):
                        res += w1[k] * t[k](y[i - 1], y[i], x, i)
                for l in range(n_state_features):
                    for i in range(n_position):
                        res += w2[l] * s[l](y[i], x, i)
                p1[xi][yi] = pow(math.e, res)
            total = sum(p1[xi][yi] for yi in range(n_y))
            if total > 0:
                for yi in range(n_y):
                    p1[xi][yi] /= total

        if not change:
            break

        p0 = p1

    ans = {}
    for xi in range(n_x):
        for yi in range(n_y):
            ans[(tuple(x_list[xi]), y_list[yi])] = p0[xi][yi]
    return w1 + w2, ans


if __name__ == "__main__":
    # ---------- 《统计学习方法》例11.4 ----------
    def t1(y0, y1, x, i):
        return int(y0 in {0} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)} and i in {1, 2})


    def t2(y0, y1, x, i):
        return int(y0 in {0} and y1 in {0} and x in {(1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {1})


    def t3(y0, y1, x, i):
        return int(y0 in {1} and y1 in {0, 1} and x in {(0, 0, 0), (1, 1, 1)} and i in {2})


    def t4(y0, y1, x, i):
        return int(y0 in {1} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1),
                                                     (1, 0, 0), (1, 0, 1)} and i in {2})


    def t5(y0, y1, x, i):
        return int(y0 in {0, 1} and y1 in {0} and x in {(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)} and i in {1, 2})


    def s1(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1, 2})


    def s2(y0, x, i):
        return int(y0 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0),
                                       (1, 0, 1)} and i in {0})


    def s3(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1})


    def s4(y0, x, i):
        return int(y0 in {1} and x in {(1, 0, 1), (0, 1, 0)} and i in {0, 2})


    w1 = [1, 0.6, 1.2, 0.2, 1.4]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.2, 0.8, 0.5]
    s = [s1, s2, s3, s4]

    # 生成随机模型
    X, Y = make_hidden_sequence(
        w1, t, w2, s,
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])

    w, P = improved_iterative_scaling(X, Y, t, s)
    print("学习结果:", [round(elem, 2) for elem in w])
    # 生成训练数据集权重: [1, 0.6, 1.2, 0.2, 1.4, 1, 0.2, 0.8, 0.5]
    # 学习结果: [1.07, 0.75, 0.75, 0.35, 1.38, 1.04, 0.22, 0.67, 0.4] (迭代次数:613)
