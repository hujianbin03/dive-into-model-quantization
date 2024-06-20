import numpy as np


def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - np.round(x.max() / scale)
    return scale, z


def saturete(x, int_max, int_min):
    return np.clip(x, int_min, int_max)


def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturete(np.round(x / scale + z), int_max, int_min)
    return xq


def dequant_data(xq, scale, z):
    x = ((xq - z) * scale).astype('float32')
    return x


if __name__ == '__main__':
    np.random.seed(1)
    data_float32 = np.random.randn(3).astype('float32')
    int_max = 127
    int_min = -128
    print(f"input = {data_float32}")

    # 1. 计算scale
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print(f"scale = {scale}")
    print(f"z = {z}")

    # 2. 量化和截断
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print(f"quant_result = {data_int8}")
    
    # 3. 反量化
    data_dequant_float = dequant_data(data_int8, scale, z)
    print(f"dequant_result = {data_dequant_float}")

    print(f"diff = {data_dequant_float - data_float32}")


































