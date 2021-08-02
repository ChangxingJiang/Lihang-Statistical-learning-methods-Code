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


def direct_sampling_method(distribution, n_samples, a=-1e5, b=1e5, tol=1e-6, random_state=0):
    """直接抽样法抽取样本

    :param distribution: 定义分布函数的概率分布
    :param n_samples: 样本数
    :param a: 左侧边界
    :param b: 右侧边界
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


if __name__ == "__main__":
    distribution = UniformDistribution(-3, 3)
    samples = direct_sampling_method(distribution, 10, -3, 3)
    print([round(v, 2) for v in samples])  # [0.29, 1.29, 0.62, 0.27, -0.46, 0.88, -0.37, 2.35, 2.78, -0.7]
