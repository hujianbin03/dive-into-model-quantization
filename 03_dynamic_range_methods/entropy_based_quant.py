import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.stats as stats

"""
其中，generator_P函数是用来生成随机激活值的函数；smooth_distribution函数则是用来平滑处理神经网络中激活值分布的
函数；threshold_distribution函数则是主要函数，用来计算熵量化中的最小KL散度阈值，它采用的是一种从右向左搜索的策略，
对激活值分布进行分组，并通过KL散度来度量每个分组的误差，从而找到使得误差最小的最优分组。最后，该代码通过matplotlib.pyplot
将激活值分布和最小KL散度阈值可视化展示出来。

我们的目的是获得分布 P 的动态范围，我们是通过不断地取修改阈值，得到新的概率分布，然后计算 KL 散度值，我们将遍历整个直
方图，然后获得一个 KL 散度数组，获取数组中最小的 KL 散度所对应的阈值即我们想要的结果。

"""


def generator_P(size):
    walk = []
    # random.uniform: 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    avg = random.uniform(3.000, 600.999)
    std = random.uniform(500.000, 1024.959)
    for _ in range(size):
        # random.gauss(u, sigma): 生成均值为u,标准差为sigma的满足高斯分布的值
        walk.append(random.gauss(avg, std))
    return walk


def smooth_distribution(p, eps=0.0001):
    """
    用于平滑一个离散概率论分布，使得p中为0的元素=eps, 其他元素减去 eps * n_zeros / n_nonzeros，保持总值不变
    :param p:
    :param eps:
    :return:
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    # 它平滑输入分布p，通过将小值(eps * is_zeros)添加到每个零元素中，并从每个非零元素中减去一个scaled版本的这个值（-eps1 * is_nonzeros）
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


"""
threshold_distribution() 核心函数的简要说明如下：
1.首先，我们将输入数据分成一定数量的 bins(本例为2048)
2.然后，我们选定一个阈值位置(本例选的是128)，计算从该位置开始的后面所有 bin 内数据的概率之和，这个和就是所谓的 outlier count ，即被认为是离散群点的数据个数
3.接下来，我们将 outlier count 加到 reference distribution P 中，得到新的概率分布，并且用这个新的概率分布来计算 KL 散度
确保生成的概率分布是一个合法的概率分布
减小量化误差造成的影响
4.计算完 KL 散度后，我们将阈值位置向后移动一个位置，重复以上步骤，直到计算完所有可能的阈值位置，得到了一个 KL 散度数组
5.最后，我们找到 KL 散度数组中最小的那个值，即最小的 KL 散度，并记录对应的阈值位置，这个位置就是 threshold_value

重点，如何通过p分布，找到q分布。q的分布是一定的，即量化后的比特位数。p分布是可变的，所以会出现三种情况
1. p和q bin相同
2. p和q bin不同，且能整除
3. p和q bin不同，不能整除

实现：
假设input_p=[1,0,2,3,5,6] dst_bins=4
1.计算stride
stride = input.size / bin 取整 = 1

2.按照stride划分
[1] [0] [2] [3] [5](多余位) [6](多余位)

3.判断p分布每一位是否非零
[1,0,1,1,1,1]

4.将多余位累加到最后整除的位置上，在上面多余位是[5]和[6]，最后整除的位置上是[3]，因此[5+6+3=14]进行替换
[1] [0] [2] [14]

5.进行位扩展从而得到output_q
将4的结果和3的非零位进行一个映射得到最终的结果
[1] [0] [2] [4.67] [4.67] [4.67]
"""


def threshold_distribution(distribution, target_bin):
    distribution = distribution[1:]
    length = distribution.size  # 获取概率分布的大小
    threshold_sum = sum(distribution[target_bin:])  # 计算概率分布从target_bin位置开始的累加和，即outliers_count
    kl_divergence = np.zeros(length - target_bin)   # 初始化以恶搞numpy数组，用来存放每个阀值下计算得到的KL散度

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum   # 将后面outliers_count加到reference_distribution_P中，得到新的概率分布
        threshold_sum = threshold_sum - distribution[threshold]     # 更新threshold_sum的值

        is_nonzeros = (p != 0).astype(np.int64)     # 判断每一位是否非0

        quantized_bins = np.zeros(target_bin, dtype=np.int64)

        num_merged_bins = sliced_nd_hist.size // target_bin   # 计算stride

        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()   # 将多余位累加到最后整除的位置上

        q = np.zeros(sliced_nd_hist.size, dtype=np.int64)   # 进行位扩展
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j] / float(norm))

        # 平滑处理，保证KL计算出来不会无限放大
        p = smooth_distribution(p)
        q = smooth_distribution(q)

        # 计算p和q的KL散度
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    # np.argmin: 返回序列中最小值对应的索引
    min_kl_divergence = np.argmin(kl_divergence)    # 选择最小的KL散度
    threshold_value = min_kl_divergence + target_bin
    # print((min_kl_divergence + 0.5) * 8)
    return threshold_value


if __name__ == '__main__':
    # 获取KL最小阀值
    # size = 20480
    # # 生成一个随机的概率分布P
    # P = generator_P(size)
    # P = np.array(P)
    # P = P[P>0]
    data_float = np.random.randn(20480).astype(np.float32)
    data_float = data_float[data_float>0]
    # np.absolute(P): 返回每个元素的绝对值
    # print("最大的激活值 ", max(np.absolute(P)), P.size)

    # np.histogram: 是一个生成直方图的函数
    # 返回值
    # hist：一个长度为bins的一维数组，表示每个区间中数据点的数量或者归一化后的概率密度值。
    # bin_edges：长度为bins + 1的一维数组，表示每个区间的边界。
    hist, bins = np.histogram(data_float, bins=2048)
    print(len(hist), len(bins))
    threshold = threshold_distribution(hist, target_bin=128)
    print("threshold 所在组:", threshold)
    print("threshold 所在组的区间范围:", bins[threshold])
    # 分成split_zie组，density表示是否要normed
    plt.title("Relu activation value Histogram")
    plt.xlabel("Activation values")
    plt.ylabel("Normalized number of Counts")
    plt.hist(data_float, bins=2047)
    plt.vlines(bins[threshold], 0, 30, colors='r', linestyles='dashed')
    plt.show()

    print(f'input: {data_float[:10]}')
    # 求scale
    scale = bins[threshold] / 128
    print(f'scale: {scale}')
    # 量化
    quant = np.clip(np.round(data_float / scale), -128, 127)
    print(f'quant: {quant[:10]}')
    # 反量化
    dequant = (quant * scale).astype(np.float32)
    print(f'dequant: {dequant[:10]}')