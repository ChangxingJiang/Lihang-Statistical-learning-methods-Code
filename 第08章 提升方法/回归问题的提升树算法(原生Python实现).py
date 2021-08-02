from copy import copy

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class AdaBoostRegressor:
    """AdaBoost算法

    :param X: 输入变量
    :param Y: 输出变量
    :param weak_reg: 基函数
    :param M: 基函数的数量
    """

    def __init__(self, X, Y, weak_reg, M=10):
        self.X, self.Y = X, Y
        self.weak_reg = weak_reg
        self.M = M

        self.n_samples = len(self.X)
        self.G_list = []  # 基函数列表

        # ---------- 执行训练 ----------
        self._train()

    def _train(self):
        # 计算当前的残差：f(x)=0时
        r = [self.Y[i] for i in range(self.n_samples)]

        # 迭代增加基函数
        for m in range(self.M):
            # print("平方损失函数:", sum(c * c for c in r), "残差:", r)

            # 拟合残差学习一个基函数
            self.weak_reg.fit(self.X, r)

            self.G_list.append(copy(self.weak_reg))

            # 计算更新后的新残差
            predict = self.weak_reg.predict(self.X)
            for i in range(self.n_samples):
                r[i] -= predict[i]

    def predict(self, x):
        """预测实例"""
        return sum(self.G_list[i].predict([x])[0] for i in range(len(self.G_list)))


if __name__ == "__main__":
    # ---------- 《统计学习方法》例8.2 ----------
    dataset = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
               [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]]

    seg = AdaBoostRegressor(dataset[0], dataset[1], DecisionTreeRegressor(max_depth=1), M=6)
    r = sum((seg.predict(dataset[0][i]) - dataset[1][i]) ** 2 for i in range(10))
    print("平方误差损失:", r)

    # ---------- sklearn波士顿房价 ----------
    X, Y = load_boston(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    seg = AdaBoostRegressor(x1, y1, DecisionTreeRegressor(max_depth=1), M=50)
    r = sum((seg.predict(x2[i]) - y2[i]) ** 2 for i in range(len(x2)))
    print("平方误差损失:", r)
