import torch

data = torch.tensor([[0.6, 0, 3, 4], [0, 0, 8, 9.2]])
zero_count_per_batch = torch.sum(data == 0, dim=1).squeeze(0)
print(zero_count_per_batch)
