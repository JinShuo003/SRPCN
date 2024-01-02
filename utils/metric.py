import open3d as o3d

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


def chamfer_distance(pcd1, pcd2):
    """
    Chamfer Distance.

    Args:
        pcd1 (torch.tensor): (B, N, 3)
        pcd2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = ChamferDistance(pcd1, pcd2)
    return dist1, dist2


def earth_movers_distance(pcd1, pcd2):
    """
    Earth Mover's Distance.

    Args:
        pcd1 (torch.tensor): (B, N, 3)
        pcd2 (torch.tensor): (B, N, 3)
    """
    dists = EarthMoverDistance(pcd1, pcd2)
    return dists


def f_score(pred, gt, th=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        th   (float): a distance threshhold
    """
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0


def symmetry_cd_l1(pcd1, pcd2):
    dist1, dist2 = chamfer_distance(pcd1, pcd2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (dist1 + dist2) / 2.0


def symmetry_cd_l2(pcd1, pcd2):
    dist1, dist2 = chamfer_distance(pcd1, pcd2)
    return (dist1 + dist2) / 2.0


