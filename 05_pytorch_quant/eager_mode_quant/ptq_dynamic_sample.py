import torch
from torch import nn
from torch.ao.quantization import PerChannelMinMaxObserver, MinMaxObserver

from utils import print_size_of_model
from sample_net import SampleNet


def test_dynamic_sample_net():
    net = SampleNet()
    print(f'原始网络： {net}')
    for i in net.named_modules():
        if isinstance(i[1], nn.Linear):
            print(f'Linear.weight: {i[1].weight.data}')

    q_s = torch.ao.quantization.QConfig(activation=PerChannelMinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
    ), weight=MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
    ))

    q_net = torch.quantization.quantize_dynamic(net, {nn.Linear: q_s})
    print(f'量化之后网络：{q_net}')
    # named_modules: 查看模型中每个层的名称和对应的模块对象
    print(f'量化之后每层网络结构: ')
    q_linear_w = None
    for i in q_net.named_modules():
        if isinstance(i[1], torch.ao.nn.quantized.dynamic.modules.linear.Linear):
            q_linear_w = i[1].weight()
        print(i)
    # 说明调用了torch.quantization.quantize_dynamic，linear.weight，已经量化完成了
    # 并且新加了一层网络结构fc._packed_params，记录量化后weight等信息
    print(f'Linear.weight: {q_linear_w}')

    print('-' * 48 + '测试前向传播' + '-' * 48)
    input_t = torch.Tensor([[[[-1, -2, -3], [1, 2, 3]]]])
    net(input_t)

    print('-' * 50 + '量化之后' + '-' * 50)
    q_net(input_t)

    print('-' * 50 + 'size' + '-' * 50)
    print(f'原网络size: ')
    print_size_of_model(net)
    print(f'量化后网络size：')
    print_size_of_model(q_net)


if __name__ == '__main__':
    torch.manual_seed(111)
    torch.backends.quantized.engine = 'qnnpack'
    test_dynamic_sample_net()
