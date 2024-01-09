from utils.loss import ibs_loss, cd_loss_L1
import torch
import numpy as np
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D


if __name__ == "__main__":
    center = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    radius = np.array([[[2], [1], [3], [4]]])
    pcd = np.array([[[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.5], [0, 0, 0]]])
    center = torch.from_numpy(np.asarray(center).astype(np.float32)).cuda()
    radius = torch.from_numpy(np.asarray(radius).astype(np.float32)).cuda()
    pcd = torch.from_numpy(np.asarray(pcd).astype(np.float32)).cuda()
    print(center.shape)
    print(radius.shape)
    print(pcd.shape)

    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, _, idx1, _ = cham_loss(center, pcd)
    loss = torch.abs(torch.sqrt(dist1) - radius)
    print(dist1)
    print(torch.sqrt(dist1).shape)
    print(idx1)
    print(loss)

