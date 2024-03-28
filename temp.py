import torch


input = torch.tensor([[1, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=torch.float32)
torch.where(torch.norm(input))