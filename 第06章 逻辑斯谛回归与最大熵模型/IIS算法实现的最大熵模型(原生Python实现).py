import math

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


def improved_iterative_scaling(x, y, features, error=1e-6):
    """改进的迭代尺度法求最大熵模型

    :param x: 输入变量
    :param y: 输出变量
    :param features: 特征函数列表
    :return:
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
    n_total = n_x * n_y  # 不同样本的总数

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

    # 定义w的初值和模型P(Y|X)的初值
    w0 = [0] * n_features  # w的初值：wi=0
    p0 = [[1 / n_total] * n_y for _ in range(n_x)]  # 当wi=0时，P(Y|X)的值

    change = True
    while change:
        change = False

        # 遍历各个特征条件以更新w
        for j in range(n_features):
            def func(d, jj):
                """目标方程"""
                res = 0
                for xxi in range(n_x):
                    for yyi in range(n_y):
                        if features[jj](list(x_list[xxi]), y_list[yyi]):
                            res += d2[xxi] * p0[xxi][yyi] * pow(math.e, d * nn[xxi][yyi])
                res -= d3[jj]
                return res

            # 牛顿法求解目标方程的根
            dj = newton_method_linear(func, args=(j,))

            # 更新wi的值
            w0[j] += dj
            if abs(dj) >= error:
                change = True

        # 计算新的模型
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

        if not change:
            ans = {}
            for xi in range(n_x):
                for yi in range(n_y):
                    ans[(tuple(x_list[xi]), y_list[yi])] = p1[xi][yi]
            return w0, ans

        p0 = p1


if __name__ == "__main__":
    dataset = [[[1], [1], [1], [1], [2], [2], [2], [2]], [1, 2, 2, 3, 1, 1, 1, 1]]


    def f1(xx, yy):
        return xx == [1] and yy == 1


    def f2(xx, yy):
        return (xx == [1] and yy == 2) or (xx == [1] and yy == 3)


    # {((1,), 1): 0.2500000558794252, ((1,), 2): 0.3749999720602874, ((1,), 3): 0.3749999720602874, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
    print(improved_iterative_scaling(dataset[0], dataset[1], [f1]))

    # {((1,), 1): 0.24999946967330844, ((1,), 2): 0.3750002651633458, ((1,), 3): 0.3750002651633458, ((2,), 1): 0.3333333333333333, ((2,), 2): 0.3333333333333333, ((2,), 3): 0.3333333333333333}
    print(improved_iterative_scaling(dataset[0], dataset[1], [f2]))
