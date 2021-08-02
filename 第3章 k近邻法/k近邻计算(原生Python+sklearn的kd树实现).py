from collections import Counter

from sklearn.neighbors import KDTree


class KDTreeKNN:
    """kd实现的k近邻计算"""

    def __init__(self, x, y, k, metric="euclidean"):
        self.x, self.y, self.k = x, y, k
        self.kdtree = KDTree(self.x, metric=metric)  # 构造KD树

    def count(self, x):
        """计算实例x所属的类y"""
        index = self.kdtree.query([x], self.k, return_distance=False)
        count = Counter()
        for i in index[0]:
            count[self.y[i]] += 1
        return count.most_common(1)[0][0]


if __name__ == "__main__":
    dataset = [[(3, 3), (4, 3), (1, 1)], [1, 1, -1]]  # 训练数据集
    knn = KDTreeKNN(dataset[0], dataset[1], k=2)
    print(knn.count((3, 4)))  # 1
