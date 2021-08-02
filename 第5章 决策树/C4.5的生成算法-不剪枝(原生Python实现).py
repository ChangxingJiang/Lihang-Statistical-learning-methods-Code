from collections import Counter
from collections import defaultdict
from math import log

import numpy as np


def entropy(y, base=2):
    """计算随机变量Y的熵"""
    count = Counter(y)
    ans = 0
    for freq in count.values():
        prob = freq / len(y)
        ans -= prob * log(prob, base)
    return ans


def conditional_entropy(x, y, base=2):
    """计算随机变量X给定的条件下随机变量Y的条件熵H(Y|X)"""
    freq_y_total = defaultdict(Counter)  # 统计随机变量X取得每一个取值时随机变量Y的频数
    freq_x = Counter()  # 统计随机变量X每一个取值的频数
    for i in range(len(x)):
        freq_y_total[x[i]][y[i]] += 1
        freq_x[x[i]] += 1
    ans = 0
    for xi, freq_y_xi in freq_y_total.items():
        res = 0
        for freq in freq_y_xi.values():
            prob = freq / freq_x[xi]
            res -= prob * log(prob, base)
        ans += res * (freq_x[xi] / len(x))
    return ans


class DecisionTreeID3WithoutPruning:
    """ID3生成算法构造的决策树（仅支持离散型特征）-不包括剪枝"""

    class Node:
        def __init__(self, mark, use_feature=None, children=None):
            if children is None:
                children = {}
            self.mark = mark
            self.use_feature = use_feature  # 用于分类的特征
            self.children = children  # 子结点

        @property
        def is_leaf(self):
            return len(self.children) == 0

    def __init__(self, x, y, labels=None, base=2, epsilon=0):
        if labels is None:
            labels = ["特征{}".format(i + 1) for i in range(len(x[0]))]
        self.labels = labels  # 特征的标签
        self.base = base  # 熵的单位（底数）
        self.epsilon = epsilon  # 决策树生成的阈值

        # ---------- 构造决策树 ----------
        self.n = len(x[0])
        self.root = self._build(x, y, set(range(self.n)))  # 决策树生成

    def _build(self, x, y, spare_features_idx):
        """根据当前数据构造结点

        :param x: 输入变量
        :param y: 输出变量
        :param spare_features_idx: 当前还可以使用的特征的下标
        """
        freq_y = Counter(y)

        # 若D中所有实例属于同一类Ck，则T为单结点树，并将Ck作为该结点的类标记
        if len(freq_y) == 1:
            return self.Node(y[0])

        # 若A为空集，则T为单结点树，并将D中实例数最大的类Ck作为该结点的标记
        if not spare_features_idx:
            return self.Node(freq_y.most_common(1)[0][0])

        # 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        best_feature_idx, best_gain = -1, 0
        for feature_idx in spare_features_idx:
            gain = self.information_gain(x, y, feature_idx)
            if gain > best_gain:
                best_feature_idx, best_gain = feature_idx, gain

        # 如果Ag的信息增益小于阈值epsilon，则置T为单结点树，并将D中实例数最大的类Ck作为该结点的类标记
        if best_gain <= self.epsilon:
            return self.Node(freq_y.most_common(1)[0][0])

        # 依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点
        node = self.Node(freq_y.most_common(1)[0][0], use_feature=best_feature_idx)
        features = set()
        sub_x = defaultdict(list)
        sub_y = defaultdict(list)
        for i in range(len(x)):
            feature = x[i][best_feature_idx]
            features.add(feature)
            sub_x[feature].append(x[i])
            sub_y[feature].append(y[i])

        for feature in features:
            node.children[feature] = self._build(sub_x[feature], sub_y[feature],
                                                 spare_features_idx - {best_feature_idx})
        return node

    def __repr__(self):
        """深度优先搜索绘制可视化的决策树"""

        def dfs(node, depth=0, value=""):
            if node.is_leaf:  # 处理叶结点的情况
                res.append(value + " -> " + node.mark)
            else:
                if depth > 0:  # 处理中间结点的情况
                    res.append(value + " :")
                for val, child in node.children.items():
                    dfs(child, depth + 1, "  " * depth + self.labels[node.use_feature] + " = " + val)

        res = []
        dfs(self.root)
        return "\n".join(res)

    def information_gain(self, x, y, idx):
        """计算信息增益"""
        return entropy(y, base=self.base) - conditional_entropy([x[i][idx] for i in range(len(x))], y, base=self.base)


class DecisionTreeC45WithoutPruning(DecisionTreeID3WithoutPruning):
    """C4.5生成算法构造的决策树（仅支持离散型特征）-不包含剪枝"""

    def information_gain(self, x, y, idx):
        """重写计算信息增益的方法，改为计算信息增益比"""
        return super().information_gain(x, y, idx) / entropy([x[i][idx] for i in range(len(x))], base=self.base)


if __name__ == "__main__":
    X, Y = [np.array([["青年", "否", "否", "一般"],
                      ["青年", "否", "否", "好"],
                      ["青年", "是", "否", "好"],
                      ["青年", "是", "是", "一般"],
                      ["青年", "否", "否", "一般"],
                      ["中年", "否", "否", "一般"],
                      ["中年", "否", "否", "好"],
                      ["中年", "是", "是", "好"],
                      ["中年", "否", "是", "非常好"],
                      ["中年", "否", "是", "非常好"],
                      ["老年", "否", "是", "非常好"],
                      ["老年", "否", "是", "好"],
                      ["老年", "是", "否", "好"],
                      ["老年", "是", "否", "非常好"],
                      ["老年", "否", "否", "一般", "否"]]),
            np.array(["否", "否", "是", "是", "否",
                      "否", "否", "是", "是", "是",
                      "是", "是", "是", "是", "否"])]
    decision_tree = DecisionTreeC45WithoutPruning(X, Y, labels=["年龄", "有工作", "有自己的房子", "信贷情况"])
    print(decision_tree)
