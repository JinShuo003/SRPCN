import torch

from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils.loss import emdModule


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
    return (dist1 + dist2)/2


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