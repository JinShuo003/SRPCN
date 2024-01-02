import open3d as o3d
import torch
import numpy as np

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


def l2_cd(pcd1, pcd2):
    dist1, dist2 = CD(pcd1, pcd2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return (dist1 + dist2)/2


def l1_cd(pcd1, pcd2):
    dist1, dist2 = CD(pcd1, pcd2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return (dist1 + dist2) / 2


def emd(pcd1, pcd2):
    dists = EMD(pcd1, pcd2)
    return dists


def f_score(pcd1, pcd2, threshold=0.0001):
    dist1, dist2 = CD(pcd1, pcd2)
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