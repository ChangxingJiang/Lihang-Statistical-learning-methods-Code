import numpy as np


def get_word_document_matrix(D):
    """依据TF-IDF构造的单词-文本矩阵

    :param D: 文本集合
    :return: 依据TF-IDF的单词-文本矩阵
    """
    n_samples = len(D)

    # 构造所有文本出现的单词的集合
    W = set()
    for d in D:
        W |= set(d)

    # 构造单词列表及单词下标映射
    W = sorted(W)
    mapping = {w: i for i, w in enumerate(W)}

    n_features = len(W)

    # 计算：单词出现在文本中的频数/文本中出现的所有单词的频数之和
    X = np.zeros((n_features, n_samples))
    for i, d in enumerate(D):
        for w in d:
            X[mapping[w], i] += 1
        X[:, i] /= len(d)

    # 计算：包含单词的文本数/文本集合D的全部文本数
    df = np.zeros(n_features)
    for d in D:
        for w in set(d):
            df[mapping[w]] += 1

    # 构造单词-文本矩阵
    for i in range(n_features):
        X[i, :] *= np.log(n_samples / df[i])

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

    np.set_printoptions(precision=2)
    print(get_word_document_matrix(D))

    # [[0.   0.   0.38 0.5  0.   0.   0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.3  0.   0.   0.3 ]
    #  [0.   0.75 0.   0.   0.   0.   0.   0.5  0.  ]
    #  [0.   0.   0.   0.   0.   0.   0.5  0.   0.3 ]
    #  [0.38 0.   0.   0.   0.   0.3  0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
    #  [0.38 0.   0.38 0.   0.   0.   0.   0.   0.  ]
    #  [0.   0.   0.   0.   0.   0.   0.5  0.   0.3 ]
    #  [0.   0.   0.   0.   0.   0.6  0.   0.   0.3 ]
    #  [0.27 0.   0.27 0.   0.   0.   0.   0.37 0.  ]
    #  [0.   0.   0.   0.5  0.75 0.   0.   0.   0.  ]]
