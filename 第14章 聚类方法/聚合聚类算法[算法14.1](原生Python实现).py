import numpy as np


class DSU:
    def __init__(self, n: int):
        self._n = n
        self._array = [i for i in range(n)]
        self._size = [1] * n
        self._group_num = n

    def find(self, i: int) -> int:
        """查询i所在的连通分支:O(1)"""
        if self._array[i] != i:
            self._array[i] = self.find(self._array[i])
        return self._array[i]

    def union(self, i: int, j: int) -> bool:
        """合并i和j所属的连通分支:O(1)"""
        i, j = self.find(i), self.find(j)
        if i != j:
            self._group_num -= 1
            if self._size[i] >= self._size[j]:
                self._array[j] = i
                self._size[i] += self._size[j]
            else:
                self._array[i] = j
                self._size[j] += self._size[i]
            return True
        else:
            return False

    def is_connected(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)

    @property
    def array(self) -> list:
        return self._array

    @property
    def group_num(self) -> int:
        """计算连通分支数量:O(1)"""
        return self._group_num

    @property
    def max_group_size(self) -> int:
        """计算最大连通分支包含的数量:O(N)"""
        import collections
        return max(collections.Counter(self._array).values())


def single_linkage_agglomerative_clustering(D, k):
    """聚合聚类算法：合并规则为类间距离最小，类间距离为最短距离，停止条件为类的个数是k
    时间复杂度：O(N^2 log(N^2))

    :param D: 样本之间的距离矩阵
    :param k: 停止时类的个数
    :return: 每个样本所属的类别
    """
    n_samples = len(D)

    # 排序所有的样本
    sorted_distance = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sorted_distance.append((D[i][j], i, j))
    sorted_distance.sort()

    # 初始化并查集
    dsu = DSU(n_samples)
    for d, i, j in sorted_distance:
        if dsu.find(i) != dsu.find(j):
            dsu.union(i, j)
            if dsu.group_num == k:
                break

    return dsu.array


if __name__ == "__main__":
    D = np.array([[0, 7, 2, 9, 3],
                  [7, 0, 5, 4, 6],
                  [2, 5, 0, 8, 1],
                  [9, 4, 8, 0, 5],
                  [3, 6, 1, 5, 0]])

    print(single_linkage_agglomerative_clustering(D, 2))  # [2, 1, 2, 1, 2]
