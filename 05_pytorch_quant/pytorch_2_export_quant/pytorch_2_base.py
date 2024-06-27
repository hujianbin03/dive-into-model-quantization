import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    example_inputs = (torch.randn(1, 5),)
    net = M().eval()

    # 1. 捕获程序
    traced_net = capture_pre_autograd_graph(net, *example_inputs)

    # 2. 定义量化器
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    # 3. 插入观察
    prepare_net = prepare_pt2e(traced_net, quantizer)

    # 4. 转换模型
    convert_net = convert_pt2e(prepare_net)

