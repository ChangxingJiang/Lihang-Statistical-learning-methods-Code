import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

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

    # 将文档转换为词频向量：(文本集合中的第m个文本,单词集合中的第v个单词) = 第v个单词在第m个文本中的出现频数
    count_vector = CountVectorizer()
    tf = count_vector.fit_transform([" ".join(doc) for doc in example])

    # 训练LDA主题模型：n_components = 话题数量
    lda = LatentDirichletAllocation(n_components=3,  # 话题个数K
                                    learning_method="batch",  # 学习方法：batch=变分推断EM算法(默认)；online=在线变分推断EM算法
                                    random_state=0)
    doc_topic_distr = lda.fit_transform(tf)

    print("【文本-话题计数矩阵】doc_topic_distr[i] = 第i个文本的话题分布")
    print(doc_topic_distr)
    # [[0.07 0.86 0.07]
    #  [0.12 0.76 0.12]
    #  [0.07 0.86 0.07]
    #  [0.82 0.09 0.09]
    #  [0.76 0.12 0.12]
    #  [0.88 0.06 0.06]
    #  [0.09 0.09 0.82]
    #  [0.09 0.83 0.09]
    #  [0.07 0.06 0.88]]

    print("【单词-话题非规范化概率矩阵】components_[i][j] = 第i个话题生成第j个单词的未规范化的概率")
    print(lda.components_)
    # [[1.33 1.33 0.33 0.33 1.34 3.33 0.33 0.33 2.35 0.33 2.33]
    #  [1.33 0.33 2.33 0.33 1.32 4.33 2.33 0.33 0.33 3.33 0.33]
    #  [0.34 1.34 0.33 2.33 0.34 2.34 0.33 2.33 1.32 0.33 0.33]]

    print("【单词-话题概率矩阵】components_[i][j] = 第i个话题生成第j个单词的概率")
    print(lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis])
    # [[0.1  0.1  0.02 0.02 0.1  0.24 0.02 0.02 0.17 0.02 0.17]
    #  [0.08 0.02 0.14 0.02 0.08 0.26 0.14 0.02 0.02 0.2  0.02]
    #  [0.03 0.11 0.03 0.2  0.03 0.2  0.03 0.2  0.11 0.03 0.03]]
