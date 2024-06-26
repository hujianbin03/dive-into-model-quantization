import torch
from torch import fx
from torch.fx import symbolic_trace


# Simple module for demonstration
class MyModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)


def transform(m: torch.nn.Module,
              tracer_class: type = fx.Tracer) -> torch.nn.Module:
    graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    return fx.GraphModule(m, graph)


if __name__ == '__main__':
    # 1. Symbolic tracing(符号追踪)：提取模型内部结构(captures the semantics of the module)
    module = MyModule()

    # 符号追踪这个模块
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced = symbolic_trace(module)

    # 中间表示
    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)
    # 打印模型
    print(symbolic_traced.code)
    # 打印表
    symbolic_traced.graph.print_tabular()

    # 将加法修改为乘法
    new_model = transform(module)
    print(new_model)
