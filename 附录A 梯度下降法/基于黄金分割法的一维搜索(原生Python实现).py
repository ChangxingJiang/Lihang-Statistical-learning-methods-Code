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


if __name__ == "__main__":
    print(golden_section_for_line_search(lambda x: x ** 2, -10, 5, epsilon=1e-6))  # 5.263005013597177e-06
