from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def build_best_knn_simple_cross_validation(x, y):
    """简单交叉验证计算k最优的KNN分类器"""
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=0)  # 拆分训练集&验证集(80%)和测试集(20%)
    x11, x12, y11, y12 = train_test_split(x1, y1, test_size=0.25, random_state=0)  # 拆分训练集(60%)和验证集(20%)
    best_k, best_score = 0, 0
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x11, y11)
        score = knn.score(x12, y12)
        if score > best_score:
            best_k, best_score = k, score
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x1, y1)
    return best_k, best_knn.score(x2, y2)


if __name__ == "__main__":
    # 生成随机样本数据
    X, Y = make_blobs(n_samples=1000,
                      n_features=10,
                      centers=5,
                      cluster_std=5000,
                      center_box=(-10000, 10000),
                      random_state=0)

    # 计算k最优的KNN分类器
    final_k, final_score = build_best_knn_simple_cross_validation(X, Y)

    print("最优k:", final_k)  # 75
    print("最优k的测试集准确率:", final_score)  # 0.900
