from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

if __name__ == "__main__":
    boston = load_boston()
    X = boston.data
    Y = boston.target

    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    clf = DecisionTreeRegressor(ccp_alpha=0.16, random_state=0)
    clf.fit(x1, y1)

    print(export_text(clf, feature_names=list(boston.feature_names)))

    print("平方误差:", clf.score(x2, y2))
