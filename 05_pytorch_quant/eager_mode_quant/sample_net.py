import torch
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub


class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        # 这里输出，为了看下size变化，可以改的大一些
        self.fc = nn.Linear(3, 1024, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        print(f'input: {x}')
        x = self.conv(x)
        print(f'经过卷积后, 权重为：{self.conv.weight}, \t输出为：{x}')
        x = self.fc(x)
        fc_w = self.fc.weight
        if isinstance(self.fc, torch.ao.nn.quantized.dynamic.modules.linear.Linear):
            fc_w = self.fc.weight()
        print(f'经过fc后, 权重为：{fc_w}, \t输出为：{x}')
        x = self.relu(x)
        print(f'经过relu后, 输出为：{x}')
        return x


class SampleQuantNet(nn.Module):
    def __init__(self):
        super(SampleQuantNet, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.fc = nn.Linear(3, 2, bias=False)
        self.relu = nn.ReLU(inplace=False)
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # 手动指定张量从浮动形式转换的位置
        x = self.quant(x)
        x = self.conv(x)
        x = self.fc(x)
        x = self.relu(x)
        # 指定反量化位置
        x = self.dequant(x)
        return x


class CustomNet(nn.Module):
    def __init__(self, q=False):
        # By turning on Q we can turn on/off the quantization
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 120, bias=False)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10, bias=False)
        self.q = q
        if q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.q:
            x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Be careful to use reshape here instead of view
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        if self.q:
            x = self.dequant(x)
        return x