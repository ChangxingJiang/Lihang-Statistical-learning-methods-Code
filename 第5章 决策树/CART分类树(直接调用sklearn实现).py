import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

if __name__ == "__main__":
    # ---------- 《统计学习方法》例5.1 ----------
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

    N = len(X)
    n = len(X[0])

    # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
    y_list = list(set(Y))
    y_mapping = {c: i for i, c in enumerate(y_list)}
    x_list = [list(set(X[i][j] for i in range(N))) for j in range(n)]
    x_mapping = [{c: i for i, c in enumerate(x_list[j])} for j in range(n)]

    for i in range(N):
        for j in range(n):
            X[i][j] = x_mapping[j][X[i][j]]
    for i in range(N):
        Y[i] = y_mapping[Y[i]]

    # ---------- sklearn鸢尾花例子 ----------
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    print(export_text(clf, feature_names=["年龄", "有工作", "有自己的房子", "信贷情况"], show_weights=True))

    iris = load_iris()
    X = iris.data
    Y = iris.target

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = DecisionTreeClassifier(ccp_alpha=0.02, random_state=0)
    clf.fit(X, Y)
    print(export_text(clf, feature_names=iris.feature_names, show_weights=True))
    print(clf.score(x2, y2))
