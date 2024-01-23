import torch

from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils.loss import emdModule
import torch.nn.functional as F


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


def ibs_angle_dist(center, pcd, direction):
    # distances = torch.cdist(center, pcd, p=2)
    # min_indices = torch.argmin(distances, dim=2)
    # closest_points = torch.gather(pcd, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))
    # direction_pred = closest_points - center
    # loss = 1 - F.cosine_similarity(direction_pred, direction, dim=2)
    # return torch.mean(loss, 1)

    distances = torch.cdist(pcd, center, p=2)
    min_indices = torch.argmin(distances, dim=2)
    closest_center = torch.gather(center, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))
    direction_pred = pcd - closest_center
    cosine_sim = F.cosine_similarity(direction_pred, direction, dim=2)
    loss = torch.clamp(-cosine_sim, min=0)

    return torch.mean(loss, 1)


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