import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class SVM:
    """支持向量机

    :param X: 输入变量列表
    :param Y: 输出变量列表
    :param C: 正则化项（惩罚参数：C越大，对误分类的惩罚越大）
    :param kernel_func: 核函数
    :param tol: 容差
    :param max_iter: 最大迭代次数
    """

    def __init__(self, X, Y, kernel_func=None, C=1, tol=1e-4, max_iter=100):
        # ---------- 检查参数 ----------
        # 检查输入变量和输出变量
        if len(X) != len(Y):
            raise ValueError("输入变量和输出变量的样本数不同")
        if len(X) == 0:
            raise ValueError("输入样本数不能为0")
        self.X, self.Y = X, Y

        # 检查正则化项
        if C <= 0:
            raise ValueError("正则化项必须严格大于0")
        self.C = C

        # 检查核函数
        if kernel_func is None:
            kernel_func = self._linear_kernel  # 当未设置核函数时默认使用线性核函数
        self.kernel_func = kernel_func

        # 检查容差
        if tol <= 0:
            raise ValueError("容差必须大于0")
        self.tol = tol

        # 检查最大迭代步数
        if max_iter <= 0:
            raise ValueError("迭代步数必须大于0")
        self.max_iter = max_iter

        # ---------- 初始化计算 ----------
        self.n_samples = len(X)  # 计算样本数
        self.n_features = len(X[0])  # 计算特征数
        self.kernel_matrix = self._count_kernel_matrix()  # 计算核矩阵

        # ---------- 取初值 ----------
        self.A = np.zeros(self.n_samples)  # 拉格朗日乘子(alpha)
        self.b = 0  # 参数b
        self.E = [float(-self.Y[i]) for i in range(self.n_samples)]  # 初始化Ei的列表

        # ---------- SMO算法训练支持向量机 ----------
        self.smo()  # SMO算法计算了拉格朗日乘子的近似解
        self.support = [i for i, v in enumerate(self.A) if v > 0]  # 计算支持向量的下标列表

    def smo(self):
        """使用序列最小最优化(SMO)算法训练支持向量机"""
        for k in range(self.max_iter):
            change_num = 0  # 更新的样本数

            for i1 in self.outer_circle():  # 外层循环：依据7.4.2.1选择第1个变量（找到a1并更新后继续向后遍历，而不回到第1个）
                i2 = next(self.inner_circle(i1))  # 内层循环：依据7.4.2.2选择第2个变量（没有处理特殊情况下用启发式规则继续寻找a2）

                a1_old, a2_old = self.A[i1], self.A[i2]
                y1, y2 = self.Y[i1], self.Y[i2]
                k11, k22, k12 = self.kernel_matrix[i1][i1], self.kernel_matrix[i2][i2], self.kernel_matrix[i1][i2]

                eta = k11 + k22 - 2 * k12  # 根据式(7.107)计算η(eta)
                a2_new = a2_old + y2 * (self.E[i1] - self.E[i2]) / eta  # 依据式(7.106)计算未经剪辑的a2_new

                # 计算a2_new所在对角线线段端点的界
                if y1 != y2:
                    l = max(0, a2_old - a1_old)
                    h = min(self.C, self.C + a2_old - a1_old)
                else:
                    l = max(0, a2_old + a1_old - self.C)
                    h = min(self.C, a2_old + a1_old)

                # 依据式(7.108)剪辑a2_new
                if a2_new > h:
                    a2_new = h
                if a2_new < l:
                    a2_new = l

                # 依据式(7.109)计算a_new
                a1_new = a1_old + y1 * y2 * (a2_old - a2_new)

                # 依据式(7.115)和式(7.116)计算b1_new和b2_new并更新b
                b1_new = -self.E[i1] - y1 * k11 * (a1_new - a1_old) - y2 * k12 * (a2_new - a2_old) + self.b
                b2_new = -self.E[i2] - y1 * k12 * (a1_new - a1_old) - y2 * k22 * (a2_new - a2_old) + self.b
                if 0 < a1_new < self.C and 0 < a2_new < self.C:
                    self.b = b1_new
                else:
                    self.b = (b1_new + b2_new) / 2

                # 更新a1,a2
                self.A[i1], self.A[i2] = a1_new, a2_new

                # 依据式(7.105)计算并更新E
                self.E[i1], self.E[i2] = self._count_g(i1) - y1, self._count_g(i2) - y2

                if abs(a2_new - a2_old) > self.tol:
                    change_num += 1

            # print("迭代次数:", k, "change_num =", change_num)

            if change_num == 0:
                break

    def predict(self, x):
        """预测实例"""
        return np.sign(sum(self.A[i] * self.Y[i] * self.kernel_func(x, self.X[i]) for i in self.support) + self.b)

    def _linear_kernel(self, x1, x2):
        """计算特征向量x1和特征向量x2的线性核函数的值"""
        return sum(x1[i] * x2[i] for i in range(self.n_features))

    def outer_circle(self):
        """外层循环生成器"""
        for i1 in range(self.n_samples):  # 先遍历所有在间隔边界上的支持向量点
            if -self.tol < self.A[i1] < self.C + self.tol and not self._satisfied_kkt(i1):
                yield i1
        for i1 in range(self.n_samples):  # 再遍历整个训练集的所有样本点
            if not -self.tol < self.A[i1] < self.C + self.tol and not self._satisfied_kkt(i1):
                yield i1

    def inner_circle(self, i1):
        """内层循环生成器：未考虑特殊情况下启发式选择a2的情况"""
        max_differ = 0
        i2 = -1
        for ii2 in range(self.n_samples):
            differ = abs(self.E[i1] - self.E[ii2])
            if differ > max_differ:
                i2, max_differ = ii2, differ
        yield i2

    def _count_kernel_matrix(self):
        """计算核矩阵"""
        kernel_matrix = [[0] * self.n_samples for _ in range(self.n_samples)]
        for i1 in range(self.n_samples):
            for i2 in range(i1, self.n_samples):
                kernel_matrix[i1][i2] = kernel_matrix[i2][i1] = self.kernel_func(self.X[i1], self.X[i2])
        return kernel_matrix

    def _count_g(self, i1):
        """依据式(7.104)计算g(x)"""
        return sum(self.A[i2] * self.Y[i2] * self.kernel_matrix[i1][i2] for i2 in range(self.n_samples)) + self.b

    def _satisfied_kkt(self, i):
        """判断是否满足KKT条件"""
        ygi = self.Y[i] * self._count_g(i)  # 计算 yi*g(xi)
        if -self.tol < self.A[i] < self.tol and ygi >= 1 - self.tol:
            return True  # (7.111)式的情况: ai=0 && yi*g(xi)>=1
        elif -self.tol < self.A[i] < self.C + self.tol and abs(ygi - 1) < self.tol:
            return True  # (7.112)式的情况: 0<ai<C && yi*g(xi)=1
        elif self.C - self.tol < self.A[i] < self.C + self.tol and ygi <= 1 + self.tol:
            return True  # (7.113)式的情况: ai=C && yi*g(xi)<=1
        else:
            return False


if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    start_time = time.time()

    svm = SVM(x1, y1)
    n1, n2 = 0, 0
    for xx, yy in zip(x2, y2):
        if svm.predict(xx) == yy:
            n1 += 1
        else:
            n2 += 1

    end_time = time.time()

    print("正确率:", n1 / (n1 + n2))
    print("运行时间:", end_time - start_time)
