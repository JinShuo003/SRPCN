import torch

from metric import *


def cd_loss_L1(pcd1, pcd2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = chamfer_distance(pcd1, pcd2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcd1, pcd2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = chamfer_distance(pcd1, pcd2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(output, gt):
    dists = earth_movers_distance(output, gt)[0]
    return torch.mean(dists)
