import numpy as np


def scale_cal(x, int_min, int_max):
    scale = (x.max() - x.min()) / (int_max - int_min)
    return scale


def saturete(x, int_max, int_min):
    return np.clip(x, int_min, int_max)


def quant_float_data(x, scale, int_min, int_max):
    xq = saturete(np.round(x / scale), int_max, int_min)
    return xq


def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x


if __name__ == '__main__':
    # 固定随机种子
    np.random.seed(1)
    data_float32 = np.random.randn(3).astype('float32')
    int_max = 127
    int_min = -128
    print(f'input = {data_float32}')

    # 1. 计算Scale
    scale = scale_cal(data_float32, int_min, int_max)
    print(f'scale = {scale}')
    # 2. 量化和截断
    data_int8 = quant_float_data(data_float32, scale, int_min, int_max)
    print(f'quant_result = {np.round(data_float32 / scale)}')
    print(f'saturete_result = {data_int8}')
    # 3. 反量化
    data_dequant_float = dequant_data(data_int8, scale)
    print(f'dequant_result = {data_dequant_float}')

    print(f'diff = {data_dequant_float - data_float32}')






















