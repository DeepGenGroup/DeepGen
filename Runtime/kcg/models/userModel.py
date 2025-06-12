import torch
import torch.nn as nn
import torch.optim as optim
from kcg.TorchInjector import *
from kcg.ModelUtils import *

# 2. 定义神经网络模型
class MoonClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super(MoonClassifier, self).__init__()
        
        # 使用 nn.Linear 的层
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        # 自定义权重参数（用于 torch.matmul）
        self.custom_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.custom_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 第一层：使用 nn.Linear
        x = self.linear1(x)
        x = self.relu(x)
        
        # 自定义层：使用 torch.matmul
        # 相当于一个无偏置的线性变换：x = x @ W
        x = torch.matmul(x, self.custom_weight)
        
        # 添加偏置项
        x = x + self.custom_bias
        x = self.relu(x)
        
        # 输出层：使用 nn.Linear
        x = self.linear2(x)
        x = self.sigmoid(x)
        
        return x
    
    def describe(self):
        """打印模型信息"""
        print("模型结构:")
        print(f"  Linear1: {self.linear1}")
        print(f"  Custom Weight: {self.custom_weight.shape}")
        print(f"  Custom Bias: {self.custom_bias.shape}")
        print(f"  Linear2: {self.linear2}")

# 3. 训练函数
def train_model(model, X_train, y_train, X_test, y_test, epochs=500, lr=0.01):
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    test_losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            # 训练集损失
            train_losses.append(loss.item())
            
            # 测试集损失
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            # 计算准确率
            predictions = (test_outputs > 0.5).float()
            accuracy = (predictions == y_test).float().mean()
            accuracies.append(accuracy.item())
        
        # 每100个epoch打印一次进度
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Test Loss: {test_loss.item():.4f}, '
                  f'Accuracy: {accuracy.item():.4f}')
    
    return train_losses, test_losses, accuracies


# 5. 主函数
def main():
    # 创建模型
    model = MoonClassifier(input_dim=2, hidden_dim=32, output_dim=1).to(7)
    # model.describe()
    x = torch.rand((16,2),dtype=torch.float32, device='cuda:7')
    out = model(x).to(7)
    print(out)
    
    model2 = get_op_optimized_model(model).to(7)
    out2 = model2(x)
    if torch.allclose(out,out2):
        print("===== test correct!")
    else:
        print("===== test error!")
    for v in OpProxy.GetCollectedKernelArgs() :
        print('collected :',v)

if __name__ == "__main__":
    main()