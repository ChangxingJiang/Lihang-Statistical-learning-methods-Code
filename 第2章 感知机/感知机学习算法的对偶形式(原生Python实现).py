def count_gram(x):
    """计算Gram矩阵

    :param x: 输入变量
    :return: 输入变量的Gram矩阵
    """
    n_samples = len(x)  # 样本点数量
    n_features = len(x[0])  # 特征向量维度数
    gram = [[0] * n_samples for _ in range(n_samples)]  # 初始化Gram矩阵

    # 计算Gram矩阵
    for i in range(n_samples):
        for j in range(i, n_samples):
            gram[i][j] = gram[j][i] = sum(x[i][k] * x[j][k] for k in range(n_features))

    return gram


def dual_form_perceptron(x, y, eta):
    """感知机学习算法的对偶形式

    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的a(alpha)和b
    """
    n_samples = len(x)  # 样本点数量
    a0, b0 = [0] * n_samples, 0  # 选取初值a0(alpha),b0
    gram = count_gram(x)  # 计算Gram矩阵

    while True:  # 不断迭代直至没有误分类点
        for i in range(n_samples):
            yi = y[i]

            val = 0
            for j in range(n_samples):
                xj, yj = x[j], y[j]
                val += a0[j] * yj * gram[i][j]

            if (yi * (val + b0)) <= 0:
                a0[i] += eta
                b0 += eta * yi
                break
        else:
            return a0, b0


if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    print(dual_form_perceptron(dataset[0], dataset[1], eta=1))  # ([2, 0, 5], -3)
