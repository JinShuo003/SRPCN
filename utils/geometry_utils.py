import open3d as o3d
import numpy as np


def get_pcd_from_np(array: np.ndarray):
    assert array.ndim == 2 and array.shape[1] == 3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)

    return pcd


def get_pcd_normalize_para(pcd):
    """
    求点云的轴对齐包围盒中心与归一化参数
    Args:
        pcd: 点云
    Returns:
        centroid: 轴对齐包围盒中心
        scale: 点云到centroid的最远距离
    """
    pcd_np = np.asarray(pcd.points)

    # 求get_axis_aligned_bounding_box
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    centroid = np.asarray(axis_aligned_bounding_box.get_center())
    scale = np.max(np.sqrt(np.sum((pcd_np-centroid) ** 2, axis=1)))
    return centroid, scale


def geometry_transform(geometry, centroid, scale):
    geometry.translate(-centroid)
    geometry.scale(1 / scale, np.array([0, 0, 0]))
    return geometry


def get_sphere_mesh(radius=1.0):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_mesh.compute_vertex_normals()
    return sphere_mesh


def get_sphere_pcd(radius=0.5, points_num=256):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_pcd = sphere_mesh.sample_points_uniformly(points_num)
    return sphere_pcd


def get_unit_coordinate(size=0.5):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coord_frame.compute_vertex_normals()
    return coord_frame



