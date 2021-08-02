from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
    clf.fit(x1, y1)
    print("预测正确率:", clf.score(x2, y2))  # 预测正确率: 0.9736842105263158
