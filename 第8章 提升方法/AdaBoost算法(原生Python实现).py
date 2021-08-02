import math
from copy import copy

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    """AdaBoost算法

    :param X: 输入变量
    :param Y: 输出变量
    :param weak_clf: 弱分类算法
    :param M: 弱分类器数量
    """

    def __init__(self, X, Y, weak_clf, M=10):
        self.X, self.Y = X, Y
        self.weak_clf = weak_clf
        self.M = M

        # ---------- 初始化计算 ----------
        self.n_samples = len(self.X)
        self.n_features = len(self.X[0])

        # ---------- 取初值 ----------
        self.G_list = []  # 基本分类器列表
        self.a_list = []  # 分类器系数列表

        # ---------- 执行训练 ----------
        self._train()

    def _train(self):
        # 初始化训练数据的权值分布
        D = [1 / self.n_samples] * self.n_samples

        # 当前所有弱分类器的线性组合的预测结果
        fx = [0] * self.n_samples

        # 迭代增加弱分类器
        for m in range(self.M):
            # 使用具有权值分布D的训练数据集学习，得到基本分类器
            self.weak_clf.fit(self.X, self.Y, sample_weight=D)

            # 使用Gm(x)预测训练数据集的所有实例点
            predict = self.weak_clf.predict(self.X)

            # 计算Gm(x)在训练数据集上的分类误差率
            error = sum(D[i] for i in range(self.n_samples) if np.sign(predict[i]) != self.Y[i])

            # 计算Gm(x)的系数
            a = 0.5 * math.log((1 - error) / error, math.e)

            self.G_list.append(copy(self.weak_clf))
            self.a_list.append(a)

            # 更新训练数据集的权值分布
            D = [D[i] * pow(math.e, -a * self.Y[i] * predict[i]) for i in range(self.n_samples)]
            Z = sum(D)  # 计算规范化因子
            D = [v / Z for v in D]

            # 计算当前所有弱分类器的线性组合的误分类点数
            wrong_num = 0
            for i in range(self.n_samples):
                fx[i] += a * predict[i]  # 累加当前所有弱分类器的线性组合的预测结果
                if np.sign(fx[i]) != self.Y[i]:
                    wrong_num += 1

            print("迭代次数:", m + 1, ";", "误分类点数:", wrong_num)

            # 如果当前所有弱分类器的线性组合已经没有误分类点，则结束迭代
            if wrong_num == 0:
                break

    def predict(self, x):
        """预测实例"""
        return np.sign(sum(self.a_list[i] * self.G_list[i].predict([x]) for i in range(len(self.G_list))))


if __name__ == "__main__":
    # ---------- 《统计学习方法》例8.1 ----------
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]]

    clf = AdaBoost(dataset[0], dataset[1], DecisionTreeClassifier(max_depth=1))
    correct = 0
    for ii in range(10):
        if clf.predict(dataset[0][ii]) == dataset[1][ii]:
            correct += 1
    print("预测正确率:", correct / 10)

    # ---------- sklearn乳腺癌数据 ----------
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoost(x1, y1, DecisionTreeClassifier(max_depth=1))
    correct = 0
    for i in range(len(x2)):
        if clf.predict(x2[i]) == y2[i]:
            correct += 1
    print("预测正确率:", correct / len(x2))
