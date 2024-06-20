import numpy as np

from dynamic_range_methods.entropy import cal_kl


def smooth_data(p, eps=0.0001):
    """
    在计算KL散度时，要保证q(i) != 0，需要加上一个很小的正数eps
    :param p:
    :param eps:
    :return:
    """
    # 获取p中非0元素和0元素
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def smooth_cal_kl(p, split_p):
    # 3. 假设p是直方图统计的频次，其概率计算直接使用p /= sum(p)
    q = []
    for arr in split_p:
        # 2. 求平均： 划分的数据 / 非0个数
        # 防止分母为0
        if np.count_nonzero(arr) == 0:
            avg = 0.
        else:
            avg = np.sum(arr) / np.count_nonzero(arr)
        print(f'avg = {avg}')
        for item in arr:
            if item != 0:
                q.append(avg)
                continue
            q.append(0)
    print(f'q = {q}')
    p /= np.sum(p)
    q /= np.sum(q)
    print(f'p-norm = {p}')
    print(f'q-norm = {q}')
    # 4. 平滑数据：使得数据在=0的时候，不为0
    p = smooth_data(p)
    q = smooth_data(q)
    print(f'p-smooth = {p}')
    print(f'q-smooth = {q}')
    # 5. 计算KL散度
    KL = cal_kl(p, q)
    print(f'KL divergence = {KL}')


def entropy_not_same_bin_divisible():
    """
    p和q分布的bin不一致，但是能整除
    bins：表示数据分成的区间数
    :return:
    """
    # p.bin = 8, q.bin = 4
    p = [0, 0, 2, 3, 5, 3, 1, 7]
    bin = 4
    # 分割p，份数为bin，当不能整除的时候，会从前往后依次多一个元素
    # 1. 数据划分
    split_p = np.array_split(p, bin)
    print(f'split_data = {split_p}')
    smooth_cal_kl(p, split_p)


def entropy_not_same_bin_not_divisible():
    """
    p和q分布的bin不一致，也不能整除
    bins：表示数据分成的区间数
    :return:
    """
    # p.bin = 6, q.bin = 4
    p = [1, 0, 2, 3, 5, 6]
    bin = 4
    print(f'input = {p} bin = {bin}')
    if len(p) < bin:
        raise 'p长度小于bin'
    # 1. 计算stride, 向下取整
    stride = int(np.size(p) / bin)
    excess = np.size(p) % bin
    print(f'stride = {stride}, excess = {excess}')
    # 2. 划分数据，将不能整除的位，加到最后一位
    split_p = np.array_split(p[:-excess], bin)
    # 获取最后一份划分的数据，转list，将余数之后的数据加到list中
    split_p[-1] = np.array(list(split_p[-1]) + p[-excess:])
    print(f'split_p = {split_p}')

    smooth_cal_kl(p, split_p)


if __name__ == '__main__':
    entropy_not_same_bin_divisible()
    # entropy_not_same_bin_not_divisible()




























































