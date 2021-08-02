from collections import Counter

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # ---------- 随机数据 ----------
    X, Y = make_blobs(n_samples=1500, n_features=1,
                      centers=[[-2], [2]], cluster_std=1, random_state=0)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    n_components = 2  # 类别数
    n_samples = len(x1)  # 样本数
    n_features = len(x1[0])  # 特征数
    n_samples_of_type = Counter(y1)  # 每个类别的样本数

    # 计算各个类别的平均值
    means = [[0] * n_features for _ in range(n_components)]
    for yi in range(n_components):
        for j in range(n_features):
            means[yi][j] = sum(x1[i][j] for i in range(n_samples) if y1[i] == yi) / n_samples_of_type[yi]

    # 训练高斯混合模型
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=0)
    gmm.fit(x1)

    # 计算高斯混合模型的每个类别对应的实际类别
    mapping = {t1: t2 for t1, t2 in enumerate(pairwise_distances_argmin(means, gmm.means_))}

    # 计算准确率
    correct = 0
    for x, actual_y in zip(x2, y2):
        predict_y = mapping[gmm.predict([x])[0]]
        if predict_y == actual_y:
            correct += 1
    print("准确率:", correct / len(x2))

    # ---------- sklearn鸢尾花数据 ----------
    X, Y = load_iris(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    n_components = 3  # 类别数
    n_samples = len(x1)  # 样本数
    n_features = len(x1[0])  # 特征数
    n_samples_of_type = Counter(y1)  # 每个类别的样本数

    # 计算各个类别的平均值
    means = [[0] * n_features for _ in range(n_components)]
    for yi in range(n_components):
        for j in range(n_features):
            means[yi][j] = sum(x1[i][j] for i in range(n_samples) if y1[i] == yi) / n_samples_of_type[yi]

    # 训练高斯混合模型
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=0)
    gmm.fit(x1)

    # 计算高斯混合模型的每个类别对应的实际类别
    mapping = {t1: t2 for t1, t2 in enumerate(pairwise_distances_argmin(means, gmm.means_))}

    # 计算准确率
    correct = 0
    for x, actual_y in zip(x2, y2):
        predict_y = mapping[gmm.predict([x])[0]]
        if predict_y == actual_y:
            correct += 1
    print("准确率:", correct / len(x2))
