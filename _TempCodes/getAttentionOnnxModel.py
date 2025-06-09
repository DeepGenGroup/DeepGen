import torch
import torch.nn as nn
import torch.onnx

class MultiHeadAttentionNetwork(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 词嵌入层
        self.embedding = nn.Embedding(1000, embed_dim)
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # PyTorch 1.9+支持
        )
        
        # 前馈层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 输入x形状: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 注意力计算
        attn_output, attn_weights = self.attention(
            query=embedded,
            key=embedded,
            value=embedded,
            need_weights=True
        )
        
        # 前馈网络
        output = self.fc(attn_output)
        return output, attn_weights

# 实例化模型
model = MultiHeadAttentionNetwork()
model.eval()  # 设置为评估模式

# 生成示例输入
batch_size = 1
seq_length = 10
dummy_input = torch.randint(0, 1000, (batch_size, seq_length))

# 导出为ONNX
output_path = "/home/xushilong/DeepGen/_TempCodes/attention_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=17,  # 推荐使用较新的opset版本
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output', 'attention_weights'],
    # dynamic_axes={
    #     'input': {0: 'batch_size', 1: 'seq_length'},  # 动态维度
    #     'output': {0: 'batch_size', 1: 'seq_length'},
    #     'attention_weights': {0: 'batch_size', 1: 'seq_length'}
    # }
)

print(f"Model exported to {output_path}")