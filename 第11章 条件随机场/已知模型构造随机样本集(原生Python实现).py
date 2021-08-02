import bisect
import math
import random


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

    X, Y = make_hidden_sequence(
        w1, t, w2, s,
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])

    for row in zip(X, Y):
        print(row)
