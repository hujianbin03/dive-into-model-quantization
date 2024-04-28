import numpy as np


def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127


def quant_float_data(x, scale):
    xq = saturete(np.round(x / scale))
    return xq


def saturete(x):
    return np.clip(x, -127, 127)


def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x


if __name__ == '__main__':
    np.random.seed(1)
    data_float32 = np.random.randn(3).astype('float32')
    print(f"input = {data_float32}")

    # 1. 计算scale
    scale = scale_cal(data_float32)
    print(f"scale = {scale}")

    # 2. 量化
    data_int8 = quant_float_data(data_float32, scale)
    print(f"quant_result = {data_int8}")

    # 3. 反量化
    data_dequant_float = dequant_data(data_int8, scale)
    print(f"dequant_result = {data_dequant_float}")

    print(f'diff = {data_dequant_float - data_float32}')

































