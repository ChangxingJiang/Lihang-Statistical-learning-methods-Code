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
