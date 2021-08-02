import math
from collections import Counter

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split


class GaussianMixture:
    """只支持1个特征的高斯混合模型能参数估计的EM算法"""

    def __init__(self, X, init_mean, n_components, max_iter=100):
        self.X = X
        self.n_components = n_components
        self.max_iter = max_iter

        self.n_samples = len(X)

        # 取参数的初始值
        self.alpha = [1 / self.n_components] * self.n_components  # 分模型权重
        self.means = init_mean  # 均值
        self.sigma = [1.0] * self.n_components  # 方差

        # 执行训练
        self._train()

    def _train(self):
        for _ in range(self.max_iter):
            # E步：根据当前模型参数，计算分模型对观测数据的响应度
            gamma = [[0] * self.n_components for _ in range(self.n_samples)]
            for j in range(self.n_samples):
                sum_ = 0
                for k in range(self.n_components):
                    gamma[j][k] = self.alpha[k] * self._count_gaussian(self.X[j], k)
                    sum_ += gamma[j][k]
                for k in range(self.n_components):
                    gamma[j][k] /= sum_

            # M步：计算新一轮迭代的模型参数
            means_new = [0.0] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                sum2 = 0  # 分母
                for j in range(self.n_samples):
                    sum1 += gamma[j][k] * self.X[j]
                    sum2 += gamma[j][k]
                means_new[k] = sum1 / sum2

            sigma_new = [1.0] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                sum2 = 0  # 分母
                for j in range(self.n_samples):
                    sum1 += gamma[j][k] * math.pow(self.X[j] - self.means[k], 2)
                    sum2 += gamma[j][k]
                sigma_new[k] = sum1 / sum2

            alpha_new = [1 / self.n_components] * self.n_components
            for k in range(self.n_components):
                sum1 = 0  # 分子
                for j in range(self.n_samples):
                    sum1 += gamma[j][k]
                alpha_new[k] = sum1 / self.n_samples

            self.alpha = alpha_new
            self.means = means_new
            self.sigma = sigma_new

    def _count_gaussian(self, x, k):
        """计算高斯密度函数"""
        return math.pow(math.e, -math.pow(x - self.means[k], 2) / (2 * self.sigma[k])) / (
            math.sqrt(2 * math.pi * self.sigma[k]))

    def predict(self, x):
        best_k, best_g = -1, 0
        for k in range(self.n_components):
            g = self.alpha[k] * self._count_gaussian(x, k)
            if g > best_g:
                best_k, best_g = k, g
        return best_k


if __name__ == "__main__":

    X, Y = make_blobs(n_samples=1500, n_features=1,
                      centers=[[-2], [2]], cluster_std=1, random_state=0)

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)
    x1 = [float(elem[0]) for elem in x1]
    x2 = [float(elem[0]) for elem in x2]

    n_components = 2  # 类别数
    n_samples = len(x1)  # 样本数
    n_samples_of_type = Counter(y1)  # 每个类别的样本数

    # 计算各个类别的平均值
    means = [[0] for _ in range(n_components)]
    for yi in range(n_components):
        means[yi][0] = sum(x1[i] for i in range(n_samples) if y1[i] == yi) / n_samples_of_type[yi]

    # 训练高斯混合模型
    gmm = GaussianMixture(x1, [0.1, 0.2], n_components)

    # 计算高斯混合模型的每个类别对应的实际类别
    mapping = {t1: t2 for t1, t2 in enumerate(pairwise_distances_argmin(means, [[m] for m in gmm.means]))}

    # 计算准确率
    correct = 0
    for x, actual_y in zip(x2, y2):
        predict_y = mapping[gmm.predict(x)]
        if predict_y == actual_y:
            correct += 1
    print("准确率:", correct / len(x2))
