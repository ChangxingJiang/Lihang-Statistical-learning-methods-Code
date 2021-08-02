from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, Y = load_boston(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    seg = GradientBoostingRegressor(n_estimators=50, learning_rate=1, max_depth=1, random_state=0, loss='ls')
    seg.fit(x1, y1)
    r = sum((seg.predict([x2[i]])[0] - y2[i]) ** 2 for i in range(len(x2)))
    print("平方误差损失:", r)
