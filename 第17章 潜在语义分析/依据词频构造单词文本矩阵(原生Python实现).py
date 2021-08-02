import numpy as np


def get_word_document_matrix(D):
    """依据词频构造的单词-文本矩阵

    :param D: 文本集合
    :return: 依据词频的单词-文本矩阵
    """
    n_samples = len(D)

    # 构造所有文本出现的单词的集合
    W = set()
    for d in D:
        W |= set(d)

    # 构造单词列表及单词下标映射
    W = sorted(W)  # 按字母顺序排序单词列表(例11.2.2中操作，可选项)
    mapping = {w: i for i, w in enumerate(W)}

    n_features = len(W)

    # 构造文本的单词词频
    X = np.zeros((n_features, n_samples))
    for i, d in enumerate(D):
        for w in d:
            X[mapping[w], i] += 1

    # 构造单词-文本矩阵
    return X


if __name__ == "__main__":
    D = [["guide", "investing", "market", "stock"],
         ["dummies", "investing"],
         ["book", "investing", "market", "stock"],
         ["book", "investing", "value"],
         ["investing", "value"],
         ["dads", "guide", "investing", "rich", "rich"],
         ["estate", "investing", "real"],
         ["dummies", "investing", "stock"],
         ["dads", "estate", "investing", "real", "rich"]]

    print(get_word_document_matrix(D))

    # [[0. 0. 1. 1. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 1. 0. 0. 1.]
    #  [0. 1. 0. 0. 0. 0. 0. 1. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 1.]
    #  [1. 0. 0. 0. 0. 1. 0. 0. 0.]
    #  [1. 1. 1. 1. 1. 1. 1. 1. 1.]
    #  [1. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 1.]
    #  [0. 0. 0. 0. 0. 2. 0. 0. 1.]
    #  [1. 0. 1. 0. 0. 0. 0. 1. 0.]
    #  [0. 0. 0. 1. 1. 0. 0. 0. 0.]]
