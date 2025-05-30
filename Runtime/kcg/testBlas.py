import torch

# 设置计算设备为 GPU（如果可用），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim = 16384*2
# for i in range(5):
# 生成两个随机的 1024x1024 的半精度矩阵，并将其移动到计算设备
matrix_a = torch.rand(dim, dim, dtype=torch.float32, device=device)
matrix_b = torch.rand(dim, dim, dtype=torch.float32, device=device)

# 进行矩阵乘法运算
result_matrix = torch.matmul(matrix_a, matrix_b)
print(result_matrix)

# 打印结果矩阵的形状和部分内容
# print("Result matrix shape:", result_matrix.shape)
# print("Result matrix:")
