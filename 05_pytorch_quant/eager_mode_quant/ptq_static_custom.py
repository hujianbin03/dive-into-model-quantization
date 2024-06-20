from sample_net import CustomNet
from utils import *

if __name__ == '__main__':
    # 加载数据
    train_dataloader, test_dataloader = get_mnist_data()

    # 定义模型
    net = CustomNet(q=False)
    print(f'原始模型： {net}')

    # 训练模型
    train(net, train_dataloader)

    # 测试原始模型大小、准确率和预测速度
    o_net_size = print_size_of_model(net)
    print(f'原始模型大小：{o_net_size}')
    o_net_acc, o_net_time = time_model_evaluation(net, test_dataloader)
    print('''原始模型准确率: {0:.3f}\n预测时间 (seconds): {1:.1f}'''.format(o_net_acc, o_net_time))

    # 定义量化模型，并复制权重
    q_net = CustomNet(q=True)
    load_model(q_net, net)
    print(f'量化模型： {q_net}')

    # 1. 融合模型
    fuse_modules(q_net)
    print(f'融合后模型： {q_net}')
    f_net_size = print_size_of_model(q_net)
    print(f'融合后模型大小：{f_net_size}')
    f_net_acc, f_net_time = time_model_evaluation(q_net, test_dataloader)
    print('''融合后模型准确率: {0:.3f}\n预测时间 (seconds): {1:.1f}'''.format(f_net_acc, f_net_time))

    # 2. 设置量化配置
    q_net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print(f'量化配置：{q_net.qconfig}')

    # 3. 插入观察
    torch.quantization.prepare(q_net, inplace=True)
    print(f'插入观察后，qnet.conv1：{q_net.conv1}')

    # 4. 模型校准，喂数据(这里直接使用训练数据)
    test(q_net, train_dataloader)
    print('模型校准完成！')
    print(f'模型校准之后的模型：{q_net}')

    # 5. 转换模型
    # 需要设量化后端，不然会报错
    torch.backends.quantized.engine = 'qnnpack'
    torch.quantization.convert(q_net, inplace=True)
    print('模型转换(量化)完成！')
    print(f'量化完成之后的模型：{q_net}')
    print(f'量化完成后，qnet.conv1：{q_net.conv1}')

    q_net_size = print_size_of_model(q_net)
    print(f'量化完成后模型大小：{q_net_size}')
    q_net_acc, q_net_time = time_model_evaluation(q_net, test_dataloader)
    print('''量化完成后模型准确率: {0:.3f}\n预测时间 (seconds): {1:.1f}'''.format(q_net_acc, q_net_time))

    size_reduce = o_net_size - q_net_size
    acc_reduce = o_net_acc - q_net_acc
    time_reduce = o_net_time - q_net_time
    print('量化之后，模型大小减少：{0:.3f}(MB), 减少百分比：{1:.2f}%'.format(size_reduce, 100 * size_reduce / o_net_size))
    print('量化之后，模型准确率减少：{0:.3f}%, 减少百分比：{1:.2f}%'.format(acc_reduce, 100 * acc_reduce / o_net_acc))
    print('量化之后，模型预测时间减少：{0:.3f}s, 减少百分比：{1:.2f}%'.format(time_reduce, 100 * time_reduce / o_net_time))


































