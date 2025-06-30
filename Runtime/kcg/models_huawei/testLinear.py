import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重参数 (shape: [out_features, in_features])
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 初始化偏置参数 (shape: [out_features])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        # 使用与 torch.nn.Linear 相同的初始化方法
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # 使用 torch.matmul 进行矩阵乘法
        output = torch.matmul(x, self.weight.t())
        
        # 添加偏置
        if self.bias is not None:
            output += self.bias
        
        return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

# 测试代码
if __name__ == "__main__":
    import math
    
    # 创建原始 Linear 层和自定义实现
    torch.manual_seed(42)
    linear_layer = nn.Linear(3, 5)
    
    torch.manual_seed(42)
    custom_linear = CustomLinear(3, 5)
    
    # 创建测试输入
    x = torch.randn(2, 3)  # batch_size=2, in_features=3
    
    # 前向传播比较
    output_original = linear_layer(x)
    output_custom = custom_linear(x)
    
    # 检查输出是否接近
    print("输出差异:", torch.allclose(output_original, output_custom, atol=1e-6))
    
    # 检查参数是否相同
    print("权重差异:", torch.allclose(linear_layer.weight, custom_linear.weight))
    print("偏置差异:", torch.allclose(linear_layer.bias, custom_linear.bias))