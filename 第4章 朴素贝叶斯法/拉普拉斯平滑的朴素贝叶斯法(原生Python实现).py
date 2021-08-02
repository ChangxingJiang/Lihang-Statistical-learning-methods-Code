class NaiveBayesAlgorithmWithSmoothing:
    """拉普拉斯平滑的朴素贝叶斯法（仅支持离散型数据）

    使用列表存储先验概率和条件概率
    """

    def __init__(self, x, y, l=1):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数
        self.l = l  # 贝叶斯估计的lambda参数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        self.table1 = [0] * self.K
        for i in range(self.N):
            self.table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        self.table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                self.table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = (self.table1[k] + self.l) / (self.N + self.l * self.K)

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = (self.table2[j][k][t] + self.l) / (self.table1[k] + self.l * self.Sj[j])

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= self.l / (self.table1[k] + self.l * self.Sj[j])
            print(self.y_list[k], ":", score)
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y


if __name__ == "__main__":
    # 例4.2
    dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
    naive_bayes = NaiveBayesAlgorithmWithSmoothing(*dataset)
    print(naive_bayes.predict([2, "S"]))
