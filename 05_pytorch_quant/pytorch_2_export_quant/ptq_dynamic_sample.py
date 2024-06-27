from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from utils import *
from sample_net import SampleAlexNet


if __name__ == '__main__':
    # 加载数据
    train_dataloader, test_dataloader = get_cifar10_data()

    # 定义模型，并训练或者加载
    model = SampleAlexNet()
    device = "cpu"
    get_sample_alex_net(model, train_dataloader)
    print(f'原始模型： {model}')

    # 测试原始模型大小、准确率和预测速度
    o_net_size = print_size_of_model(model)
    print(f'原始模型大小：{o_net_size}')
    o_net_acc, o_net_time = time_model_evaluation(model, test_dataloader)
    print('''原始模型准确率: {0:.3f}\n预测时间 (seconds): {1:.1f}'''.format(o_net_acc, o_net_time))

    # 定义量化模型，并复制权重
    q_model = SampleAlexNet()
    load_model(q_model, model)
    # 对于训练后的量化，我们需要将模型设置为eval模式
    q_model.eval()

    # 1. 获得一个输入例子
    example_inputs = (next(iter(train_dataloader))[0],)

    # 2. 捕获程序
    model_traced = capture_pre_autograd_graph(q_model, example_inputs)

    # 3. 定义量化器
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))

    # 4. 插入观察
    model_prepared = prepare_pt2e(model_traced, quantizer)
    print(f'插入观察后，net_prepared.graph：{model_prepared.graph}')

    # 当我们只有动态/仅加权量化时，不需要校准
    # 5. 转换模型
    model_quantized_dynamic = convert_pt2e(model_prepared)
    print('模型转换(量化)完成！')
    print(f'量化完成之后的模型：{model_quantized_dynamic}')

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




































































