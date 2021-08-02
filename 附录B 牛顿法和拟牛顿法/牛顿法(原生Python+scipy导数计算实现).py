import numpy as np

from scipy.misc import derivative


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


def get_hessian(func, x0, dx=1e-6):
    """计算n元函数在某点的黑塞矩阵

    :param func: [function] n元函数
    :param x0: [list/tuple] 目标点的自变量坐标
    :param dx: [int/float] 计算时x的增量
    :return: [list] 黑塞矩阵
    """
    n_features = len(x0)
    ans = [[0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        def f1(xi, x1):
            x2 = list(x1)
            x2[i] = xi
            return func(x2)

        for j in range(n_features):
            # 当x[j]=xj时，x[i]方向的斜率
            def f2(xj):
                x1 = list(x0)
                x1[j] = xj
                res = derivative(f1, x0=x1[i], dx=dx, args=(x1,))
                return res

            ans[i][j] = derivative(f2, x0[j], dx=dx)

    return ans


def newton_method(func, n_features, epsilon=1e-6, maximum=1000):
    """牛顿法

    :param func: [function] n元目标函数
    :param n_features: [int] 目标函数元数
    :param epsilon: [int/float] 学习精度
    :param maximum: [int] 最大学习次数
    :return: [list] 结果点坐标
    """
    x0 = [0] * n_features  # 取初始点x0

    for _ in range(maximum):
        # 计算梯度 gk
        nabla = partial_derivative(func, x0)
        gk = np.matrix([nabla])

        # 当梯度的模长小于精度要求时，停止迭代
        if pow(sum([nabla[i] ** 2 for i in range(n_features)]), 0.5) < epsilon:
            return x0

        # 计算黑塞矩阵
        hessian = np.matrix(get_hessian(func, x0))

        # 计算步长 pk
        pk = - (hessian ** -1) * gk.T

        # 迭代
        for j in range(n_features):
            x0[j] += float(pk[j][0])


if __name__ == "__main__":
    # [0]
    print(newton_method(lambda x: x[0] ** 2, 1, epsilon=1e-6))

    # [-2.998150057576512, -3.997780069092481]
    print(newton_method(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, 2, epsilon=1e-6))
