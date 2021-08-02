def original_form_of_perceptron(x, y, eta):
    """感知机学习算法的原始形式

    :param x: 输入变量
    :param y: 输出变量
    :param eta: 学习率
    :return: 感知机模型的w和b
    """
    n_samples = len(x)  # 样本点数量
    n_features = len(x[0])  # 特征向量维度数
    w0, b0 = [0] * n_features, 0  # 选取初值w0,b0

    while True:  # 不断迭代直至没有误分类点
        for i in range(n_samples):
            xi, yi = x[i], y[i]
            if yi * (sum(w0[j] * xi[j] for j in range(n_features)) + b0) <= 0:
                w1 = [w0[j] + eta * yi * xi[j] for j in range(n_features)]
                b1 = b0 + eta * yi
                w0, b0 = w1, b1
                break
        else:
            return w0, b0


if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    print(original_form_of_perceptron(dataset[0], dataset[1], eta=1))
