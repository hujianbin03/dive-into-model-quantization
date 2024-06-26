from torch.ao.quantization import (
    get_default_qat_qconfig_mapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
from utils import *
from sample_net import SampleAlexNet


if __name__ == '__main__':
    # 加载数据
    train_dataloader, test_dataloader = get_cifar10_data()

    # 定义模型，因为qat需要训练，这里不使用模型文件加载
    net = SampleAlexNet()
    device = "cpu"
    print(f'原始模型： {net}')

    # 测试原始模型大小、准确率和预测速度
    o_net_size = print_size_of_model(net)
    print(f'原始模型大小：{o_net_size}')
    o_net_acc, o_net_time = time_model_evaluation(net, test_dataloader)
    print('''原始模型准确率: {0:.3f}\n预测时间 (seconds): {1:.1f}'''.format(o_net_acc, o_net_time))

    # 1. 设置量化配置QConfigMapping
    qconfig_mapping = get_default_qat_qconfig_mapping('qnnpack')
    # 开启训练模型
    net.train()

    # 2. 获得一个输入例子
    example_inputs = next(iter(train_dataloader))[0]

    # 3. 插入观察
    model_prepared = quantize_fx.prepare_qat_fx(net, qconfig_mapping, example_inputs)
    print(f'插入观察后，model_prepared.layer1：{model_prepared.layer1}')

    # 4. 训练
    train(net, train_dataloader)

    # 4. 转换模型
    torch.backends.quantized.engine = 'qnnpack'
    model_quantized_dynamic = quantize_fx.convert_fx(model_prepared)
    print('模型转换(量化)完成！')
    print(f'量化完成之后的模型：{model_quantized_dynamic}')
    # print(f'量化完成后，qnet.conv1：{model_quantized_dynamic.}')

    q_net_size = print_size_of_model(model_quantized_dynamic)
    print(f'量化完成后模型大小：{q_net_size}')
    q_net_acc, q_net_time = time_model_evaluation(model_quantized_dynamic, test_dataloader)
    print('''量化完成后模型准确率: {0:.3f}\n预测时间 (seconds): {1:.1f}'''.format(q_net_acc, q_net_time))

    size_reduce = o_net_size - q_net_size
    acc_reduce = o_net_acc - q_net_acc
    time_reduce = o_net_time - q_net_time
    print('量化之后，模型大小减少：{0:.3f}(MB), 减少百分比：{1:.2f}%'.format(size_reduce, 100 * size_reduce / o_net_size))
    print('量化之后，模型准确率减少：{0:.3f}%, 减少百分比：{1:.2f}%'.format(acc_reduce, 100 * acc_reduce / o_net_acc))
    print('量化之后，模型预测时间减少：{0:.3f}s, 减少百分比：{1:.2f}%'.format(time_reduce, 100 * time_reduce / o_net_time))





































