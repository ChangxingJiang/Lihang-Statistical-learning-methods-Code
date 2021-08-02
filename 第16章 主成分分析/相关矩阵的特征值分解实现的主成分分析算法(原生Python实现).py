import numpy as np


def pca_by_feature(R, need_accumulative_contribution_rate=0.75):
    """协方差矩阵/相关矩阵求解主成分及其因子载荷量和贡献率（打印到控制台）

    :param R: 协方差矩阵/相关矩阵
    :param need_accumulative_contribution_rate: 需要达到的累计方差贡献率
    :return: None
    """
    n_features = len(R)

    # 求解相关矩阵的特征值和特征向量
    features_value, features_vector = np.linalg.eig(R)

    # 依据特征值大小排序特征值和特征向量
    z = [(features_value[i], features_vector[:, i]) for i in range(n_features)]
    z.sort(key=lambda x: x[0], reverse=True)
    features_value = [z[i][0] for i in range(n_features)]
    features_vector = np.hstack([z[i][1][:, np.newaxis] for i in range(n_features)])

    # 计算所需的主成分数量
    total_features_value = sum(features_value)  # 特征值总和
    need_accumulative_contribution_rate *= total_features_value
    n_principal_component = 0  # 所需的主成分数量
    accumulative_contribution_rate = 0
    while accumulative_contribution_rate < need_accumulative_contribution_rate:
        accumulative_contribution_rate += features_value[n_principal_component]
        n_principal_component += 1

    # 输出单位特征向量和主成分的方差贡献率
    print("【单位特征向量和主成分的方差贡献率】")
    for i in range(n_principal_component):
        print("主成分:", i,
              "方差贡献率:", features_value[i] / total_features_value,
              "特征向量:", features_vector[:, i])

    # 计算各个主成分的因子载荷量：factor_loadings[i][j]表示第i个主成分对第j个变量的相关关系，即因子载荷量
    factor_loadings = np.vstack(
        [[np.sqrt(features_value[i]) * features_vector[j][i] / np.sqrt(R[j][j]) for j in range(n_features)]
         for i in range(n_principal_component)]
    )

    # 输出主成分的因子载荷量和贡献率
    print("【主成分的因子载荷量和贡献率】")
    for i in range(n_principal_component):
        print("主成分:", i, "因子载荷量:", factor_loadings[i])
    print("所有主成分对变量的贡献率:", [np.sum(factor_loadings[:, j] ** 2) for j in range(n_features)])


if __name__ == "__main__":
    X = np.array([[1, 0.44, 0.29, 0.33],
                  [0.44, 1, 0.35, 0.32],
                  [0.29, 0.35, 1, 0.60],
                  [0.33, 0.32, 0.60, 1]])

    pca_by_feature(X)

    # 【单位特征向量和主成分的方差贡献率】
    # 主成分: 0 方差贡献率: 0.542541266192316 特征向量: [-0.45990769 -0.4763124  -0.52874972 -0.53106981]
    # 主成分: 1 方差贡献率: 0.21775136378517398 特征向量: [-0.56790937 -0.49090704  0.47557056  0.45860862]
    # 【主成分的因子载荷量和贡献率】
    # 主成分: 0 因子载荷量: [-0.6775121  -0.70167866 -0.77892661 -0.78234444]
    # 主成分: 1 因子载荷量: [-0.5300166  -0.45815211  0.44383894  0.42800876]
    # 所有主成分对变量的贡献率: [0.7399402466295502, 0.7022563027546291, 0.8037196579404534, 0.7952543125853273]
