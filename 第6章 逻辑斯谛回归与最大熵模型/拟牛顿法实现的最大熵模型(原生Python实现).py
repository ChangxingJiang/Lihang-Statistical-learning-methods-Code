import math

import numpy as np
from scipy.misc import derivative


def golden_section_for_line_search(func, a0, b0, epsilon):
    """一维搜索极小值点（黄金分割法）

    :param func: [function] 一元函数
    :param a0: [int/float] 目标区域左侧边界
    :param b0: [int/float] 目标区域右侧边界
    :param epsilon: [int/float] 精度
    """
    a1, b1 = a0 + 0.382 * (b0 - a0), b0 - 0.382 * (b0 - a0)
    fa, fb = func(a1), func(b1)

    while b1 - a1 > epsilon:
        if fa <= fb:
            b0, b1, fb = b1, a1, fa
            a1 = a0 + 0.382 * (b0 - a0)
            fa = func(a1)
        else:
            a0, a1, fa = a1, b1, fb
            b1 = b0 - 0.382 * (b0 - a0)
            fb = func(b1)

    return (a1 + b1) / 2


def partial_derivative(func, arr, dx=1e-6):
    """计算n元函数在某点各个自变量的梯度向量（偏导数列表）

    :param func: [function] n元函数
    :param arr: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 偏导数
    """
    n_features = len(arr)
    ans = []
    for i in range(n_features):
        def f(x):
            arr2 = list(arr)
            arr2[i] = x
            return func(arr2)

        ans.append(derivative(f, arr[i], dx=dx))
    return ans


def bfgs_algorithm_for_maximum_entropy_model(x, y, features, error=1e-6, distance=20, maximum=1000):
    """最大熵模型学习的BFGS算法

    :param x: 输入变量
    :param y: 输出变量
    :param features: 特征函数列表
    :param error: [int/float] 学习精度
    :param distance: [int/float] 每次一维搜索的长度范围（distance倍梯度的模）
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    n_samples = len(x)  # 样本数量
    n_features = len(features)  # 特征函数数量

    # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
    y_list = list(set(y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = list(set(tuple(x[i]) for i in range(n_samples)))
    x_mapping = {c: i for i, c in enumerate(x_list)}

    n_x = len(x_list)  # 不同的x的数量
    n_y = len(y_list)  # 不同的y的数量

    print(x_list, x_mapping)
    print(y_list, y_mapping)

    # 计算联合分布的经验分布:P(X,Y) (empirical_joint_distribution)
    d1 = [[0.0] * n_y for _ in range(n_x)]  # empirical_joint_distribution
    for i in range(n_samples):
        d1[x_mapping[tuple(x[i])]][y_mapping[y[i]]] += 1 / n_samples
    print("联合分布的经验分布:", d1)

    # 计算边缘分布的经验分布:P(X) (empirical_marginal_distribution)
    d2 = [0.0] * n_x  # empirical_marginal_distribution
    for i in range(n_samples):
        d2[x_mapping[tuple(x[i])]] += 1 / n_samples
    print("边缘分布的经验分布", d2)

    # 计算特征函数关于经验分布的期望值:EP(fi) (empirical_joint_distribution_each_feature)
    # 所有特征在(x,y)出现的次数:f#(x,y) (samples_n_features)
    d3 = [0.0] * n_features  # empirical_joint_distribution_each_feature
    nn = [[0.0] * n_y for _ in range(n_x)]  # samples_n_features
    for j in range(n_features):
        for xi in range(n_x):
            for yi in range(n_y):
                if features[j](list(x_list[xi]), y_list[yi]):
                    d3[j] += d1[xi][yi]
                    nn[xi][yi] += 1

    print("特征函数关于经验分布的期望值:", d3)
    print("所有特征在(x,y)出现的次数:", nn)

    def func(ww):
        """目标函数"""
        res = 0
        for xxi in range(n_x):
            t1 = 0
            for yyi in range(n_y):
                t2 = 0
                for jj in range(n_features):
                    if features[jj](list(x_list[xxi]), y_list[yyi]):
                        t2 += ww[jj]
                t1 += pow(math.e, t2)
            res += d2[xxi] * math.log(t1, math.e)

        for xxi in range(n_x):
            for yyi in range(n_y):
                t3 = 0
                for jj in range(n_features):
                    if features[jj](list(x_list[xxi]), y_list[yyi]):
                        t3 += ww[jj]
                res -= d1[xxi][yyi] * t3

        return res

    # 定义w的初值和B0的初值
    w0 = [0] * n_features  # w的初值：wi=0
    B0 = np.identity(n_features)  # 构造初始矩阵G0(单位矩阵)

    for k in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, w0)

        g0 = np.matrix([nabla]).T  # g0 = g_k

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < error:
            break

        # 计算pk
        if k == 0:
            pk = - B0 * g0  # 若numpy计算逆矩阵时有0，则对应位置会变为inf
        else:
            pk = - (B0 ** -1) * g0

        # 一维搜索求lambda_k
        def f(xx):
            """pk 方向的一维函数"""
            x2 = [w0[jj] + xx * float(pk[jj][0]) for jj in range(n_features)]
            return func(x2)

        lk = golden_section_for_line_search(f, 0, distance, epsilon=1e-6)  # lk = lambda_k

        # print(k, "lk:", lk)

        # 更新当前点坐标
        w1 = [w0[j] + lk * float(pk[j][0]) for j in range(n_features)]

        # print(k, "w1:", w1)

        # 计算g_{k+1}，若模长小于精度要求时，则停止迭代
        # 计算新的模型

        nabla = partial_derivative(func, w1)

        g1 = np.matrix([nabla]).T  # g0 = g_{k+1}

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < error:
            w0 = w1
            break

        # 计算G_{k+1}
        yk = g1 - g0
        dk = np.matrix([[lk * float(pk[j][0]) for j in range(n_features)]]).T

        B1 = B0 + (yk * yk.T) / (yk.T * dk) + (B0 * dk * dk.T * B0) / (dk.T * B0 * dk)

        B0 = B1
        w0 = w1

    p1 = [[0.0] * n_y for _ in range(n_x)]
    for xi in range(n_x):
        for yi in range(n_y):
            for j in range(n_features):
                if features[j](list(x_list[xi]), y_list[yi]):
                    p1[xi][yi] += w0[j]
            p1[xi][yi] = pow(math.e, p1[xi][yi])
        total = sum(p1[xi][yi] for yi in range(n_y))
        if total > 0:
            for yi in range(n_y):
                p1[xi][yi] /= total

    ans = {}
    for xi in range(n_x):
        for yi in range(n_y):
            ans[(tuple(x_list[xi]), y_list[yi])] = p1[xi][yi]
    return w0, ans


if __name__ == "__main__":
    dataset = [[[1], [1], [1], [1], [2], [2], [2], [2]], [1, 2, 2, 3, 1, 1, 1, 1]]


    def f1(xx, yy):
        return xx == [1] and yy == 1


    def f2(xx, yy):
        return (xx == [1] and yy == 2) or (xx == [1] and yy == 3)


    # {((1,), 1): 0.2500000558794252, ((1,), 2): 0.3749999720602874, ((1,), 3): 0.3749999720602874, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
    print(bfgs_algorithm_for_maximum_entropy_model(dataset[0], dataset[1], [f1]))

    # {((1,), 1): 0.24999946967330844, ((1,), 2): 0.3750002651633458, ((1,), 3): 0.3750002651633458, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
    print(bfgs_algorithm_for_maximum_entropy_model(dataset[0], dataset[1], [f2]))
