import torch

from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils.loss import emdModule
import torch.nn.functional as F
import numpy as np


def medial_axis_surface_dist(center, radius, pcd):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, _, _, _ = cham_loss(center, pcd)
    dist = torch.abs(torch.sqrt(dist1) - radius)

    return torch.mean(dist, 1)


def medial_axis_interaction_dist(center, radius, pcd):
    B, N, _ = center.shape
    B, M, _ = pcd.shape

    distances = torch.cdist(center, pcd, p=2.0)
    radius = torch.unsqueeze(radius, dim=2)
    radius = radius.repeat(1, 1, M)

    loss = torch.where(radius > distances, radius - distances, 0)
    return torch.mean(loss, dim=[1, 2])

    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    # dist1, _, _, _ = cham_loss(center, pcd)
    # dist1 = torch.sqrt(dist1)
    # loss = torch.where(radius > dist1, radius - dist1, 0)

    # return torch.mean(loss, 1)


def ibs_angle_dist(center, radius, direction, pcd):
    B, N, _ = center.shape
    distances = torch.cdist(pcd, center, p=2)
    min_indices = torch.argmin(distances, dim=2)
    closest_center = center[torch.arange(B).unsqueeze(1), min_indices, :]
    closest_direction = direction[torch.arange(B).unsqueeze(1), min_indices, :]
    closest_radius = radius[torch.arange(1).unsqueeze(1), min_indices]
    min_distances = torch.gather(distances, 2, min_indices.unsqueeze(-1)).squeeze(-1)
    direction_pred = pcd - closest_center
    direction_pred = F.normalize(direction_pred, p=2, dim=2)

    cosine_sim = F.cosine_similarity(direction_pred, closest_direction, dim=2)
    loss = -cosine_sim + np.cos(np.deg2rad(90))
    # loss = torch.where(min_distances < 1.5 * closest_radius, loss, 0)
    loss = torch.clamp(loss, min=0)
    intersect_points_num = torch.sum(loss != 0, dim=1).squeeze(0).float()  # (B)

    return torch.mean(loss, 1), intersect_points_num


def l1_cd(pcd1, pcd2):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(pcd1, pcd2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return (dist1 + dist2) / 2


def l2_cd(pcd1, pcd2):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(pcd1, pcd2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return dist1 + dist2


def emd(pcd1, pcd2):
    emd_loss = emdModule()
    dists = emd_loss(pcd1, pcd2)[0]
    return torch.mean(dists, dim=1)


def f_score(pcd1, pcd2, threshold=0.0001):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(pcd1, pcd2)
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore


# def f_score(pred, gt, th=0.01):
#     fscore = []
#     b, n, _ = pred.shape
#     for i in range(b):
#         pred_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred[i]))
#         gt_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt[i]))

#         dist1 = pred_.compute_point_cloud_distance(gt_)
#         dist2 = gt_.compute_point_cloud_distance(pred_)

#         recall = float(sum(d < th for d in dist2)) / float(len(dist2))
#         precision = float(sum(d < th for d in dist1)) / float(len(dist1))
#         fscore.append(2 * recall * precision / (recall + precision) if recall + precision else 0)
#     return np.array(fscore)