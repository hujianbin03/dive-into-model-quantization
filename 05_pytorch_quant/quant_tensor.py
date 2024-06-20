import torch


def test_quantize_per_tensor():
    """
    函数原型：torch.quantize_per_tensor(input, scale, zero_point, dtype)
    功能介绍：将Tensor内的元素“量化”操作成dtype类型(quint8, qint8, uint32)的数据。
    基本计算方式为：output[i] = (input[i] / scale) + zero_point
    :return:
    """
    # 量化
    xq = torch.quantize_per_tensor(x, scale=0.1, zero_point=1, dtype=torch.quint8)
    print(f'tensor_quant = {xq}')
    # 转int
    int_xq = xq.int_repr()
    print(f'int_quant = {int_xq}')
    # 反量化
    dxq = xq.dequantize()
    print(f'dquant = {dxq}')


def test_quantize_per_channel():
    """
    函数原型：torch.quantize_per_channel(input, scale, zero_point, axis, dtype)
        input: 要被量化的原始Float tensor.
        scale：量化的Scale，Per channel的1维数组。
        zero_points：量化参数zero-points，Per channel的1维数组。
        axis：Per-channel quantize的channel方向（0： 坐标轴X方向，1：坐标轴Y方向，2：坐标轴C方向，，以此类推tensor的维度，或者(WHC...）
    功能介绍：将Tensor内的元素“量化”操作成dtype类型(quint8, qint8, qint32)的数据。和per tensor的区别是，此函数的scale和zero_point是1维数组，每个元素代表每个channel；
    基本计算方式为：output[i] = (input[i] / scale[c]) + zero_point[c]
    注意：scale和zero_points必须是1维数组，且长度需要和input.shape[axis]相等
    :return:
    """
    scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
    zero_point = torch.tensor([1, 2, 3, 4])
    axis = 2
    assert scale.numel() == zero_point.numel() == x.shape[axis], "注意：scale和zero_points必须是1维数组，且长度需要和input.shape[axis]相等"
    # 量化
    xq = torch.quantize_per_channel(x, scales=scale, zero_points=zero_point, axis=axis, dtype=torch.quint8)
    print(f'tensor_quant = {xq}')
    # 转int
    int_xq = xq.int_repr()
    print(f'int_quant = {int_xq}')
    # 反量化
    dxq = xq.dequantize()
    print(f'dquant = {dxq}')


if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.rand(2, 3, 4, dtype=torch.float32)
    print(f'input = {x}')
    # test_quantize_per_tensor()
    test_quantize_per_channel()