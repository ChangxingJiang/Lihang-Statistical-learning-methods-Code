from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseDistribution(ABC):
    """随机变量分布的抽象基类"""

    @abstractmethod
    def pdf(self, x):
        """计算概率密度函数"""
        pass

    def cdf(self, x):
        """计算分布函数"""
        raise ValueError("未定义分布函数")


class UniformDistribution(BaseDistribution):
    """均匀分布

    :param a: 左侧边界
    :param b: 右侧边界
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a < x < self.b:
            return 1 / (self.b - self.a)
        else:
            return 0

    def cdf(self, x):
        if x < self.a:
            return 0
        elif self.a <= x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1


class GaussianDistribution(BaseDistribution):
    """高斯分布（正态分布）

    :param u: 均值
    :param s: 标准差
    """

    def __init__(self, u, s):
        self.u = u
        self.s = s

    def pdf(self, x):
        return pow(np.e, -1 * (pow(x - self.u, 2)) / 2 * pow(self.s, 2)) / (np.sqrt(2 * np.pi * pow(self.s, 2)))


def direct_sampling_method(distribution, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """直接抽样法抽取样本

    :param distribution: 定义分布函数的概率分布
    :param n_samples: 样本数
    :param a: 定义域左侧边界
    :param b: 定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    samples = []
    for _ in range(n_samples):
        y = np.random.rand()

        # 二分查找解方程：F(x) = y
        l, r = a, b
        while r - l > tol:
            m = (l + r) / 2
            if distribution.cdf(m) > y:
                r = m
            else:
                l = m

        samples.append((l + r) / 2)

    return samples


def accept_reject_sampling_method(d1, d2, c, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """接受-拒绝抽样法

    :param d1: 目标概率分布
    :param d2: 建议概率分布
    :param c: 参数c
    :param n_samples: 样本数
    :param a: 建议概率分布定义域左侧边界
    :param b: 建议概率分布定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)

    samples = []
    waiting = direct_sampling_method(d2, n_samples * 2, a=a, b=b, tol=tol, random_state=random_state)  # 直接抽样法得到建议分布的样本
    while len(samples) < n_samples:
        if not waiting:
            waiting = direct_sampling_method(d2, (n_samples - len(samples)) * 2, a, b)

        x = waiting.pop()
        u = np.random.rand()
        if u <= (d1.pdf(x) / (c * d2.pdf(x))):
            samples.append(x)

    return samples


if __name__ == "__main__":
    d1 = GaussianDistribution(0, 1)
    d2 = UniformDistribution(-3, 3)

    c = (1 / np.sqrt(2 * np.pi)) / (1 / 6)  # 计算c的最小值
    samples = accept_reject_sampling_method(d1, d2, c, 10, a=-3, b=3)
    print([round(v, 2) for v in samples])  # [0.17, -0.7, -0.37, 0.88, -0.46, 0.27, 0.62, 0.29, 0.27, 0.62]
