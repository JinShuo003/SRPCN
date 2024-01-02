import sys
sys.path.append("/home/data/jinshuo/IBPCDC/ChamferDistancePytorch")

from chamfer3D import dist_chamfer_3D
import torch
import torch.nn as nn
from torch.autograd import Function

import emd


def cd_loss_L1(pcd1, pcd2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(pcd1, pcd2)
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
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(pcd1, pcd2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(output, gt):
    emd_loss = emdModule()
    dists = emd_loss(output, gt)[0]
    return torch.mean(dists)


class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert (n == m)
        assert (xyz1.size()[0] == xyz2.size()[0])
        assert (n % 1024 == 0)
        assert (batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx,
                    unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps=0.005, iters=50):
        return emdFunction.apply(input1, input2, eps, iters)

# import torch
#
# from utils.metric import *
# from extensions.chamfer_distance.chamfer_distance import ChamferDistance
# from extensions.earth_movers_distance.emd import EarthMoverDistance
#
#
# CD = ChamferDistance()
# EMD = EarthMoverDistance()
#
#
# def cd_loss_L1(pcs1, pcs2):
#     """
#     L1 Chamfer Distance.
#
#     Args:
#         pcs1 (torch.tensor): (B, N, 3)
#         pcs2 (torch.tensor): (B, M, 3)
#     """
#     dist1, dist2 = CD(pcs1, pcs2)
#     dist1 = torch.sqrt(dist1)
#     dist2 = torch.sqrt(dist2)
#     return (torch.mean(dist1) + torch.mean(dist2)) / 2.0
#
#
# def cd_loss_L2(pcs1, pcs2):
#     """
#     L2 Chamfer Distance.
#
#     Args:
#         pcs1 (torch.tensor): (B, N, 3)
#         pcs2 (torch.tensor): (B, M, 3)
#     """
#     dist1, dist2 = CD(pcs1, pcs2)
#     return torch.mean(dist1) + torch.mean(dist2)
#
#
# def emd_loss(pcs1, pcs2):
#     """
#     EMD Loss.
#
#     Args:
#         xyz1 (torch.Tensor): (b, N, 3)
#         xyz2 (torch.Tensor): (b, N, 3)
#     """
#     dists = EMD(pcs1, pcs2)
#     return torch.mean(dists)
