import numpy as np


def cosine(X):
    """计算所有样本之间的夹角余弦矩阵"""
    n_samples = len(X[0])

    # 计算平方和
    square = [np.square(X[:, i]).sum() for i in range(n_samples)]

    # 构造夹角余弦矩阵
    D = np.identity(n_samples)  # 初始化为单位矩阵

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            xi, xj = X[:, i], X[:, j]
            numerator = (xi * xj).sum()
            denominator = np.sqrt(square[i] * square[j])
            if denominator:
                D[i][j] = D[j][i] = numerator / denominator
            else:  # 当出现零平方和时
                D[i][j] = D[j][i] = np.nan

    return D


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 5, 5],
                  [2, 0, 0, 0, 2]])

    print(cosine(X))

    # [[1.                nan 0.         0.         0.37139068]
    #  [       nan 1.                nan        nan        nan]
    #  [0.                nan 1.         1.         0.92847669]
    #  [0.                nan 1.         1.         0.92847669]
    #  [0.37139068        nan 0.92847669 0.92847669 1.        ]]
