import torch
from sample_net import SampleQuantNet


def test_sample():
    net = SampleQuantNet()
    # 必须将模型设置为eval模式，静态量化逻辑才能工作
    net.eval()
    print(f'原始网络： {net}')

    # 1. 融合网络
    torch.quantization.fuse_modules(net, [['fc', 'relu']], inplace=True)
    print(f'融合后网络： {net}')

    # 2. 设置config
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # 3. 插入观察
    torch.quantization.prepare(net, inplace=True)
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
    test_sample()









































