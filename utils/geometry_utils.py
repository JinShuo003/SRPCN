import open3d as o3d
import numpy as np


def get_pcd_normalize_para(pcd):
    pcd_np = np.asarray(pcd.points)
    # 求点云的中心
    centroid = np.mean(pcd_np, axis=0)
    # 求长轴长度
    scale = np.max(np.sqrt(np.sum(pcd_np ** 2, axis=1)))
    return centroid, scale


def geometry_transform(geometry, centroid, scale):
    geometry.translate(-centroid)
    geometry.scale(1 / scale, np.array([0, 0, 0]))
    return geometry


def get_sphere_mesh(radius=1.0):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_mesh.compute_vertex_normals()
    return sphere_mesh


def get_unit_coordinate(size=0.5):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coord_frame.compute_vertex_normals()
    return coord_frame


def get_sphere_pcd(radius=0.5, points_num=256):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_pcd = sphere_mesh.sample_points_uniformly(points_num)
    return sphere_pcd
