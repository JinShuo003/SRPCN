import torch

# 假设原始数据 tensor 的形状是 (B, n, 3)
B = 4  # Batch size
n = 10  # 假设每个 batch 中有 10 个点
m = 3   # 假设要取出 3 个点

# 创建原始数据 tensor
original_tensor = torch.randn((B, n, 3))

# 创建索引 tensor，形状为 (B, m)
index_tensor = torch.randint(0, n, size=(B, m))

print(torch.arange(B).unsqueeze(1).shape)
# 使用 advanced indexing 取出对应的点
selected_points = original_tensor[torch.arange(B).unsqueeze(1), index_tensor, :]

print("原始数据 tensor 形状:", original_tensor.shape)
print("索引 tensor 形状:", index_tensor.shape)
print("取出的点 tensor 形状:", selected_points.shape)
