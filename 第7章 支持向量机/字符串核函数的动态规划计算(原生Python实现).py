"""
时间复杂度：O(N1×N2×Length)
空间复杂度：O(N1×N2)
"""


def count_kernel_function_for_string(s1, s2, length: int, att: float):
    """计算字串长度等于n的字符串核函数的值

    :param s1: 第1个字符串
    :param s2: 第2个字符串
    :param length: 需要查找的英文字符集长度
    :param att: 衰减参数
    :return: 字符串核函数的值
    """

    # 计算字符串长度
    n1, n2 = len(s1), len(s2)

    # 定义状态矩阵：dp[l][i][j] = s1的前i个字符和s2的前j个字符中，所有长度为l的相同子序列所构成的核函数值的和
    # 因为状态转移中只会用到l和l-1，所以省略l以节约空间
    dp1 = [[1] * (n2 + 1) for _ in range(n1 + 1)]

    # 字符串的核函数的值的列表：ans[i] = 子串长度为(i+1)的字符串核函数的值
    ans = []

    # 依据子串长度进行状态转移：[1,n]
    for l in range(length):
        dp2 = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        # 定义当前子串长度的核函数的值
        res = 0

        # 进行状态转移，状态转移方程如下：
        # 当前字符相同：dp[l][i][j] = dp[l][i-1][j] * att + dp[l][i][j-1] * att - dp[l][i-1][j-1] * att * att + dp[l-1][i-1][j-1] * att * att
        # 当前字符不同：dp[l][i][j] = dp[l][i-1][j] * att + dp[l][i][j-1] * att - dp[l][i-1][j-1] * att * att
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                dp2[i][j] += dp2[i - 1][j] * att + dp2[i][j - 1] * att - dp2[i - 1][j - 1] * att * att
                if s1[i - 1] == s2[j - 1]:
                    dp2[i][j] += dp1[i - 1][j - 1] * att * att
                    res += dp1[i - 1][j - 1] * att * att  # 累加当前长度核函数的值

        dp1 = dp2
        ans.append(res)

    return ans[-1]


if __name__ == "__main__":
    print(count_kernel_function_for_string("abc", "acd", 2, 0.5))  # 0.03125
    print(count_kernel_function_for_string("fog", "fob", 2, 0.5))  # 0.0625
    print(count_kernel_function_for_string("fog", "fog", 2, 0.5))  # 0.140625
    print(count_kernel_function_for_string("fob", "fob", 2, 0.5))  # 0.140625
