import numpy as np


def check_words(D):
    """整理文本集合整理为数值化的文本的单词序列

    :param D: 文本集合
    :return: 数值化的文本的单词序列
    """
    n_components = 0  # 单词集合中单词的个数
    mapping = {}  # 单词到数值化单词的映射
    X = []
    for d in D:
        x = []
        for word in d:
            if word not in mapping:
                mapping[word] = n_components
                n_components += 1
            x.append(mapping[word])
        X.append(x)
    return np.asarray(X), n_components


def lda_gibbs(D, K, A=None, B=None, n_iter=100, random_state=0):
    """LDA吉布斯抽样算法

    :param D: 文本集合：每一行为1个文本，每一列为1个单词
    :param A: 超参数αlpha
    :param B: 超参数beta
    :param K: 话题个数
    :param n_iter: 迭代次数(直到结束燃烧期)
    :param random_state: 随机种子
    :return: 文本-话题计数矩阵(M×K矩阵),单词-话题计数矩阵(K×V矩阵),文本-话题概率矩阵(M×K矩阵),单词-话题概率矩阵(K×V矩阵)
    """
    X, n_components = check_words(D)  # 数值化的文本的单词序列；单词数(V)
    n_samples = len(D)  # 文本数(M)
    n_features = [len(X[m]) for m in range(n_samples)]  # 文本中单词的个数(N_m)

    np.random.seed(random_state)

    # 初始化超参数alpha和beta：在没有其他先验知识的情况下，可以假设向量alpha和beta的所有分量均为1
    if A is None:
        A = [1] * K
    if B is None:
        B = [1] * n_components
    A_sum = sum(A)
    B_sum = sum(B)

    # 初始化计数矩阵，设所有技术矩阵的元素的初值为0
    N_kv = np.zeros((K, n_components))  # 单词-话题矩阵(K×V矩阵)
    N_k = np.zeros(K)  # 单词-话题矩阵的边缘计数
    N_mk = np.zeros((n_samples, K))  # 文本-话题矩阵(M×K矩阵)
    N_m = np.zeros(n_samples)  # 文本-话题矩阵的边缘计数

    # 给文本的单词序列的每个位置上随机指派一个话题
    Z = [[np.random.randint(0, K) for _ in range(n_features[m])] for m in range(n_samples)]

    # 根据随机指派的话题，更新计数矩阵
    for m in range(n_samples):  # 遍历所有文本
        for n in range(n_features[m]):  # 遍历第m个文本中的所有单词
            v = X[m][n]  # 当前位置单词是第v个单词
            k = Z[m][n]  # 当前位置话题是第k个话题
            N_kv[k][v] += 1  # 增加话题-单词计数
            N_k[k] += 1  # 增加话题-单词和计数
            N_mk[m][k] += 1  # 增加文本-话题计数
            N_m[m] += 1  # 增加文本-话题和计数

    # 循环执行以下操作，直到结束燃烧期
    for _ in range(n_iter):
        for m in range(n_samples):  # 遍历所有文本
            for n in range(n_features[m]):  # 遍历第m个文本中的所有单词
                v = X[m][n]  # 当前位置单词是第v个单词
                k = Z[m][n]  # 当前位置话题是第k个话题

                # 对话题-单词矩阵和文本-话题矩阵中当期位置的已有话题的计数减1
                N_kv[k][v] -= 1  # 减少话题-单词计数
                N_k[k] -= 1  # 减少话题-单词和计数
                N_mk[m][k] -= 1  # 减少文本-话题计数
                N_m[m] -= 1  # 减少文本-话题和计数

                # 按照满条件分布进行抽样（以下用于抽样的伪概率没有除以分母）
                p = np.zeros(K)
                for k in range(K):
                    p[k] = ((N_kv[k][v] + B[v]) / (N_k[k] + B_sum)) * ((N_mk[m][k] + A[k]) / (N_m[m] + A_sum))
                p /= np.sum(p)

                k = np.random.choice(range(K), size=1, p=p)[0]

                # 对话题-单词矩阵和文本-话题矩阵中当期位置的新话题的计数加1
                N_kv[k][v] += 1  # 增加话题-单词计数
                N_k[k] += 1  # 增加话题-单词和计数
                N_mk[m][k] += 1  # 增加文本-话题计数
                N_m[m] += 1  # 增加文本-话题和计数

                # 更新文本的话题序列
                Z[m][n] = k

    # 利用得到的样本计数，计算模型参数
    T = np.zeros((n_samples, K))  # theta(M×K矩阵)
    for m in range(n_samples):
        for k in range(K):
            T[m][k] = N_mk[m][k] + A[k]
        T[m, :] /= np.sum(T[m, :])

    P = np.zeros((K, n_components))  # phi(K×V矩阵)
    for k in range(K):
        for v in range(n_components):
            P[k][v] = N_kv[k][v] + B[v]
        P[k, :] /= np.sum(P[k, :])

    return N_mk, N_kv, T, P


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    example = [["guide", "investing", "market", "stock"],
               ["dummies", "investing"],
               ["book", "investing", "market", "stock"],
               ["book", "investing", "value"],
               ["investing", "value"],
               ["dads", "guide", "investing", "rich", "rich"],
               ["estate", "investing", "real"],
               ["dummies", "investing", "stock"],
               ["dads", "estate", "investing", "real", "rich"]]

    N_mk, N_kv, T, P = lda_gibbs(example, 3)

    print("文本-话题计数矩阵(M×K矩阵):")
    print(N_mk)
    # [[1. 0. 3.]
    #  [0. 0. 2.]
    #  [4. 0. 0.]
    #  [1. 1. 1.]
    #  [0. 2. 0.]
    #  [0. 4. 1.]
    #  [0. 2. 1.]
    #  [3. 0. 0.]
    #  [1. 1. 3.]]

    print("文本-话题概率矩阵(M×K矩阵):")
    print(T)
    # [[0.29 0.14 0.57]
    #  [0.2  0.2  0.6 ]
    #  [0.71 0.14 0.14]
    #  [0.33 0.33 0.33]
    #  [0.2  0.6  0.2 ]
    #  [0.12 0.62 0.25]
    #  [0.17 0.5  0.33]
    #  [0.67 0.17 0.17]
    #  [0.25 0.25 0.5 ]]

    print("单词-话题计数矩阵(K×V矩阵):")
    print(N_kv)
    # [[0. 2. 1. 3. 1. 2. 0. 0. 1. 0. 0.]
    #  [0. 2. 0. 0. 0. 0. 2. 1. 2. 2. 1.]
    #  [2. 5. 1. 0. 1. 0. 0. 1. 0. 0. 1.]]

    print("单词-话题概率矩阵(K×V矩阵):")
    print(P)
    # [[0.05 0.14 0.1  0.19 0.1  0.14 0.05 0.05 0.1  0.05 0.05]
    #  [0.05 0.14 0.05 0.05 0.05 0.05 0.14 0.1  0.14 0.14 0.1 ]
    #  [0.14 0.27 0.09 0.05 0.09 0.05 0.05 0.09 0.05 0.05 0.09]]
