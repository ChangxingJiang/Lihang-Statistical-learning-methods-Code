from sklearn.datasets import make_blobs

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def build_best_knn_s_fold_cross_validation(dataset):
    """S折交叉验证计算k最优的KNN分类器

    :param dataset: 数据集
    :return: k最优的KNN分类器；最优的KNN分类器的预测准确率
    """
    X, Y = dataset
    best_k, best_score = 0, 0
    for k in range(1, 101):
        # 构造使用最近邻的k个实例的KNN分类器
        knn = KNeighborsClassifier(n_neighbors=k)

        # 使用S折交叉验证计算当前分类器的准确率
        scores = cross_val_score(knn, X, Y, cv=10, scoring="accuracy")
        score = scores.mean()

        # 如果当前分类器的准确率更高，则取用当前分类器
        if score > best_score:
            best_k, best_score = k, score

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X, Y)
    return best_knn, best_score


if __name__ == "__main__":
    # 生成随机样本数据
    dataset = make_blobs(n_samples=1000,
                         n_features=10,
                         centers=5,
                         cluster_std=5000,
                         center_box=(-10000, 10000),
                         random_state=0)

    # 计算k最优的KNN分类器
    best_knn, best_score = build_best_knn_s_fold_cross_validation(dataset)

    print("最优k:", best_knn.n_neighbors)  # 73
    print("最优k的测试集准确率:", best_score)  # 0.9119999999999999
