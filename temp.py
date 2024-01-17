import open3d as o3d
from utils import geometry_utils
import numpy as np
from utils.loss import ibs_angle_loss
import torch
import torch.optim as optim 


if __name__ == '__main__':
    center = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    pcd_init = torch.tensor([[[0.05, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]], dtype=torch.float32)
    pcd_init.requires_grad = True
    direction = torch.tensor([[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)

    optimizer = optim.Adam([pcd_init], lr=0.001, betas=(0.9, 0.999))
    lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    for epoch in range(10):
        print(pcd_init.grad)
        # optimizer.zero_grad()

        loss = ibs_angle_loss(center, pcd_init, direction)

        print(loss.item())

        loss.backward()
        optimizer.step()

