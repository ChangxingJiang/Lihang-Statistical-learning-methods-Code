import numpy as np


def metropolis_hastings_method(d1, func, m, n, x0, random_state=0):
    """Metroplis-Hastings算法抽取样本

    :param d1: 目标概率分布的概率密度函数
    :param func: 目标求均值函数
    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数
    :param n: 迭代步数
    :param random_state: 随机种子
    :return: 随机样本列表,随机样本的目标函数均值
    """
    np.random.seed(random_state)

    samples = []  # 随机样本列表
    sum_ = 0  # 目标求均值函数的和

    n_features = len(x0)

    # 循环执行n次迭代
    for k in range(n):
        # 按照建议分布q(x,x')随机抽取一个候选状态
        # q(x,x')为均值为x，方差为1的正态分布
        x1 = np.random.multivariate_normal(x0, np.diag([1] * n_features), 1)[0]

        # 计算接受概率
        a = min(1, d1(x1) / d1(x0))

        # 从区间(0,1)中按均匀分布随机抽取一个数u
        u = np.random.rand()

        # 若u<=a，则转移状态；否则不转移
        if u <= a:
            x0 = x1

        # 收集样本集合
        if k >= m:
            samples.append(x0)
            sum_ += func(x0)

    return samples, sum_ / (n - m)


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt


    def d1_pdf(x):
        """随机变量x=(x_1,x_2)的联合概率密度"""
        return x[0] * pow(np.e, -x[1]) if 0 < x[0] < x[1] else 0


    def f(x):
        """目标求均值函数"""
        return x[0] + x[1]


    samples, avg = metropolis_hastings_method(d1_pdf, f, m=1000, n=11000, x0=[5, 8])

    print(samples)  # [array([0.39102823, 0.58105655]), array([0.39102823, 0.58105655]), ...]
    print("样本目标函数均值:", avg)  # 4.720997790412456


    def draw_distribution():
        """绘制总体概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                Z[i][j] = d1_pdf([i / 10, j / 10])

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()


    def draw_sample():
        """绘制样本概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i, j in samples:
            if i < 10 and j < 10:
                Z[int(i // 0.1)][int(j // 0.1)] += 1

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()
