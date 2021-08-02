import heapq


class KDTree:
    class _Node:
        """kd树的轻量级结点"""
        __slots__ = "element", "axis", "left", "right"

        def __init__(self, element, axis=0, left=None, right=None):
            self.element = element  # 当前结点的值
            self.axis = axis  # 当前结点用于切分的轴
            self.left = left  # 当前结点的左子结点
            self.right = right  # 当前结点的右子结点

        def __lt__(self, other):
            """定义_Node之间的小于关系（避免heapq比较大小报错）"""
            return self.element < other.element

    def __init__(self, data, distance_func):
        """构造平衡kd树实例"""
        self._size = len(data)  # 元素总数
        self._distance_func = distance_func  # 用于计算距离的函数
        if self._size > 0:
            self._dimension = len(data[0])  # 计算输入数据的空间维度数
            self._root = self._build_kd_tree(data, depth=0)  # kd树的根结点
        else:
            self._dimension = 0
            self._root = None

    def _build_kd_tree(self, data, depth):
        """根据输入数据集data和当前深度depth，构造是平衡kd树"""
        if not data:
            return None

        # 处理当前结点数据
        select_axis = depth % self._dimension  # 计算当前用作切分的坐标轴
        median_index = len(data) // 2  # 计算中位数所在坐标
        data.sort(key=lambda x: x[select_axis])  # 依据需要用作切分的坐标轴排序输入的数据集

        # 构造当前结点
        node = self._Node(data[median_index], axis=select_axis)
        node.left = self._build_kd_tree(data[:median_index], depth + 1)  # 递归构造当前结点的左子结点
        node.right = self._build_kd_tree(data[median_index + 1:], depth + 1)  # 递归构造当前结点的右子结点
        return node

    def search_nn(self, x):
        """返回x的最近邻点"""
        return self.search_knn(x, 1)

    def search_knn(self, x, k):
        """返回距离x最近的k个点"""
        res = []
        self._search_knn(res, self._root, x, k)
        return [(node.element, -distance) for distance, node in sorted(res, key=lambda xx: -xx[0])]

    def _search_knn(self, res, node, x, k):
        if node is None:
            return

        # 计算当前结点到目标点的距离
        node_distance = self._distance_func(node.element, x)

        # 计算当前结点到目标点的距离（在当前用于划分的维度上）
        node_distance_axis = self._distance_func([node.element[node.axis]], [x[node.axis]])

        # [第1步]处理当前结点
        if len(res) < k:
            heapq.heappush(res, (-node_distance, node))
        elif node_distance_axis < (-res[0][0]):
            heapq.heappushpop(res, (-node_distance, node))

        # [第2步]处理目标点所在的子结点
        if x[node.axis] <= node.element[node.axis]:
            self._search_knn(res, node.left, x, k)
        else:
            self._search_knn(res, node.right, x, k)

        # [第3步]处理目标点不在的子结点
        if len(res) < k or node_distance_axis < (-res[0][0]):
            if x[node.axis] <= node.element[node.axis]:
                self._search_knn(res, node.right, x, k)
            else:
                self._search_knn(res, node.left, x, k)

    def __len__(self):
        """返回kd树P中元素的数量"""
        return self._size
