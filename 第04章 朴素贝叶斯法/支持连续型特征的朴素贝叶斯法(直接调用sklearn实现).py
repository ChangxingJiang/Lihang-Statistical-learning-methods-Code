from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    X, Y = load_breast_cancer(return_X_y=True)
    x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

    # 高斯朴素贝叶斯
    gnb = GaussianNB()
    gnb.fit(x1, y1)
    print(gnb.score(x2, y2))  # 0.9210526315789473

    # 多项分布朴素贝叶斯
    mnb = MultinomialNB()
    mnb.fit(x1, y1)
    print(mnb.score(x2, y2))  # 0.9105263157894737

    # 补充朴素贝叶斯
    mnb = ComplementNB()
    mnb.fit(x1, y1)
    print(mnb.score(x2, y2))  # 0.9052631578947369

    # 伯努利朴素贝叶斯
    bnb = BernoulliNB()
    bnb.fit(x1, y1)
    print(bnb.score(x2, y2))  # 0.6421052631578947
