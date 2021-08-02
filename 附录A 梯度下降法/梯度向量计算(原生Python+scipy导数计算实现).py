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


if __name__ == "__main__":
    # [6.000000000838668]
    print(partial_derivative(lambda x: x[0] ** 2, [3]))

    # [3.000000000419334, 3.9999999996709334]
    print(partial_derivative(lambda x: ((x[0] + 3) ** 2 + (x[1] + 4) ** 2) / 2, [0, 0]))
