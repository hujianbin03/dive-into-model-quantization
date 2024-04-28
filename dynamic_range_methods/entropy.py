import numpy as np
import matplotlib.pyplot as plt


def cal_kl(p, q):
    KL = 0
    for i in range(len(p)):
        KL += p[i] * np.log(p[i] / q[i])
    return KL


def kl_test(x, kl_threshold=0.01):
    # y_out = []
    while True:
        y = [np.random.uniform(1, size + 1) for i in range(size)]
        y /= np.sum(y)
        kl_result = cal_kl(x, y)
        if kl_result < kl_threshold:
            print(kl_result)
            y_out = y
            plt.plot(x)
            plt.plot(y)
            break
    return y_out


if __name__ == '__main__':
    np.random.seed(1)
    size = 10
    # np.random.uniform()作用于从一个均匀分布的区域中随机采样。
    x = [np.random.uniform(1, size + 1) for i in range(size)]
    x /= np.sum(x)
    y_out = kl_test(x, kl_threshold=0.01)
    plt.show()
    print(x, y_out)
