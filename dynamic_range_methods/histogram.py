import numpy as np
import matplotlib.pyplot as plt


def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127


"""
np.histogram是用于生成直方图的函数，其参数和返回值如下：

参数：

a：待处理的数据，可以是一维或者多维数组，多维数组将会被展开成一维数组
bins：表示数据分成的区间数
range：表示数据的取值范围，可以是一个元组或数组
density：是否将直方图归一化。如果为True，直方图将归一化为概率密度函数。默认False
weights：每个数据点的权重，可以是一维数组或与a数组相同的形状的数组。默认为None，表示所有的数据点权重相同。
返回值：

hist：一个长度为bins的一维数组，表示每个区间中数据点的数量或者归一化后的概率密度值。
bin_edges：长度为bins + 1的一维数组，表示每个区间的边界。

"""


def histogram_range(x):
    hist, range = np.histogram(x, bins=100)
    # print(hist, hist[0:99].sum() / len(x))
    total = len(x)
    left, right = 0, len(hist) - 1
    limit = 0.99
    while True:
        cover_percent = hist[left: right].sum() / total
        if cover_percent <= limit:
            break
        if hist[left] > hist[right]:
            right -= 1
        else:
            left += 1
    # max(left,right): 就是整个输入数据“去除离散值之后”的最大值
    left_val = range[left]
    right_val = range[right]
    dynamic_range = max(abs(left_val), abs(right_val))
    return dynamic_range / 127


if __name__ == '__main__':
    np.random.seed(1)
    data_float = np.random.randn(1000).astype(np.float32)
    print(f'input: {data_float[:10]}')

    scale = scale_cal(data_float)
    scale2 = histogram_range(data_float)
    print(f'scale: {scale} scale2: {scale2}')

    plt.hist(data_float, bins=100)
    plt.title("histogram")
    plt.xlabel("value")
    plt.ylabel("freq")
    plt.show()

    # 量化 float32 -> int8
    quant_data = np.clip(np.round(data_float / scale), -128, 127)
    quant_data2 = np.clip(np.round(data_float / scale), -128, 127)
    print(f'quant data: {quant_data[:10]}' f'\nquant data2: {quant_data2[:10]}')

    # 反量化
    dequant_data = (quant_data * scale).astype(np.float32)
    dequant_data2 = (quant_data2 * scale).astype(np.float32)
    print(f'dequant data: {dequant_data[:10]} \ndequant data2: {dequant_data2[:10]}')
































