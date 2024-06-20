import torch

from sample_net import SampleQuantNet


def test_qat_sample_net():
    net = SampleQuantNet()
    # 设置训练模式
    net.train()
    print(f'原始网络：{net}')

    # 1. 设置config
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    print(f'1.设置config：{net}')

    # 2. 融合模型
    torch.quantization.fuse_modules(net, [['fc', 'relu']], inplace=True)

    # 3. 插入观察
    torch.quantization.prepare_qat(net, inplace=True)
    print(f'插入观察后： {net}')

    # 4. 喂数据
    data_float = torch.rand(100, 1, 2, 3)
    for i in data_float:
        net(i)
    # 可以看到val_max, val_min已经算出来了
    print(f'喂数据之后的模型：{net}')

    # 5. 转换模型
    torch.backends.quantized.engine = 'qnnpack'
    # 转换模型之后，得到了每个op的scale和zp
    torch.quantization.convert(net, inplace=True)
    print(f'转换模型后：{net}')


if __name__ == '__main__':
    test_qat_sample_net()
