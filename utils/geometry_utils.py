import copy

import open3d as o3d
import numpy as np
import torch
import trimesh
import pyvista as pv
from scipy.spatial.transform import Rotation


def read_point_cloud(path):
    """
    读取点云
    Args:
        path: 点云文件的路径
    Returns:
        open3d.PointCloud
    """
    pcd = None
    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception as e:
        print(e.__str__())
    return pcd


def read_mesh(path):
    """
    读取Mesh
    Args:
        path: Mesh文件的路径
    Returns:
        open3d.TriangleMesh
    """
    mesh = None
    try:
        mesh = o3d.io.read_triangle_mesh(path)
    except Exception as e:
        print(e)
    return mesh


def read_medial_axis_sphere_total(path):
    """
    读取中轴球
    Args:
        path: 中轴球的路径
    Returns:
        center, radius
    """
    data = np.load(path)
    return data["center"], data["radius"], data["direction1"], data["direction2"]


def read_medial_axis_sphere_single(path):
    """
    读取归一化后的中轴球
    Args:
        path: 中轴球的路径
    Returns:
        center, radius
    """
    data = np.load(path)
    return data["center"], data["radius"], data["direction"]


def get_pcd_from_np(array: np.ndarray):
    """
    将(n, 3)的np.ndarray转为open3d.PointCloud
    Args:
        array: np.ndarray, shape: (n, 3)
    Returns:
        open3d.PointCloud
    """
    if array.ndim != 2 or array.shape[1] != 3:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)

    return pcd


def get_pcd_normalize_para(pcd):
    """
    求点云的轴对齐包围盒中心点与归一化参数
    Args:
        pcd: 点云
    Returns:
        centroid: 轴对齐包围盒中心
        scale: 点云到centroid的最远距离
    """
    pcd_o3d = pcd
    if isinstance(pcd, np.ndarray):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_np = np.asarray(pcd_o3d.points)
    axis_aligned_bounding_box = pcd_o3d.get_axis_aligned_bounding_box()
    centroid = np.asarray(axis_aligned_bounding_box.get_center())
    scale = np.max(np.sqrt(np.sum((pcd_np-centroid) ** 2, axis=1)))
    return centroid, scale


def normalize_geometry(geometry, centroid, scale):
    """
    将open3d.Geometry归一化到单位球内
    Args:
        geometry: open3d.Geometry
        centroid: Geometry的轴对齐包围盒中心点
        scale: Geometry的尺度
    Returns:
        归一化后的open3d.geometry，几何中心位于(0, 0, 0)，所有几何元素位于单位球内
    """
    geometry_copy = copy.deepcopy(geometry)
    geometry_copy.translate(-centroid)
    geometry_copy.scale(1 / scale, np.array([0, 0, 0]))
    return geometry_copy


def denormalize_geometry(geometry, centroid, scale):
    """
    将open3d.Geometry变换到原有坐标系
    Args:
        geometry: open3d.Geometry
        centroid: Geometry的轴对齐包围盒中心点
        scale: Geometry的尺度
    Returns:
        反归一化后的open3d.geometry，几何中心位于centroid，轴对齐包围盒的对角线长度为scale
    """
    geometry_copy = copy.deepcopy(geometry)
    geometry_copy.scale(scale, np.array([0, 0, 0]))
    geometry_copy.translate(centroid)
    return geometry_copy


def denormalize_geometry_tensor(geometry, centroid, scale):
    """
    将torch.Tensor变换到原有坐标系
    Args:
        geometry: torch.Tensor, (n, 3)
        centroid: Geometry的轴对齐包围盒中心点
        scale: Geometry的尺度
    Returns:
        反归一化后的torch.Tensor，几何中心位于centroid，轴对齐包围盒的对角线长度为scale
    """
    geometry *= scale
    geometry += centroid
    return geometry


def denormalize_geometry_tensor_batch(geometry, centroid, scale):
    """
    将一个batch的torch.Tensor变换到原有坐标系
    Args:
        geometry: torch.Tensor, (B, n, 3)
        centroid: Geometry的轴对齐包围盒中心点, (B, 3)
        scale: Geometry的尺度, (B)
    Returns:
        反归一化后的torch.Tensor，几何中心位于centroid，轴对齐包围盒的对角线长度为scale
    """
    geometry *= scale.unsqueeze(1).unsqueeze(2)
    geometry += centroid.unsqueeze(1)
    return geometry


def sphere_transform(center, radius, centroid, scale):
    """
    将球心为center，半径为radius的球以centroid、scale进行平移和缩放
    """
    return (center-centroid)/scale, radius/scale


def get_sphere_mesh(center=(0, 0, 0), radius=1.0):
    """
    获取半径为radius，格式为open3d.TriangleMesh的球
    Args:
        center: 球心
        radius: 球半径
    Returns:
        r=radius，format=open3d.TriangleMesh的球
    """
    sphere_mesh = pyvista2o3d(pv.Sphere(radius, center))
    return sphere_mesh


def get_sphere_pcd(center=(0, 0, 0), radius=1.0, points_num=256):
    """
    获取r=radius，格式为open3d.PointCloud的球
    Args:
        center: 球心
        radius: 球半径
        points_num: 点云点数
    Returns:
        r=radius，点数=points_num，format=open3d.PointCloud的球
    """
    sphere_mesh = get_sphere_mesh(center, radius)
    sphere_pcd = sphere_mesh.sample_points_uniformly(points_num)
    return sphere_pcd


def get_coordinate(size=1):
    """
    获取尺度为size的标准坐标轴
    Args:
        size: 坐标轴尺度
    Returns:
        尺度为size，format=open3d.TriangleMesh的标准坐标轴
    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coord_frame.compute_vertex_normals()
    return coord_frame


def get_arrow(direction=np.array([0, 0, 1]), location=np.array([0, 0, 0]), length=0.02):
    epsilon = 1e-8
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.003, cylinder_height=length/2,
                                                   cone_height=length/2)

    v = np.array([0, 0, 1])
    w = direction

    v_normalized = v / np.linalg.norm(v)
    w_normalized = w / np.linalg.norm(w)
    rotation_axis = np.cross(v_normalized, w_normalized)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    rotation_axis_norm = rotation_axis_norm if rotation_axis_norm != 0 else epsilon
    rotation_axis_normalized = rotation_axis / rotation_axis_norm
    dot_product = np.dot(v_normalized, w_normalized)
    rotation_angle = np.arccos(dot_product)

    rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis_normalized).as_matrix()

    arrow = arrow.rotate(rotation_matrix, [0, 0, 0])
    arrow.translate(location)
    return arrow


def o3d2trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return trimesh_obj


def trimesh2o3d(trimesh_obj):
    vertices = trimesh_obj.vertices
    triangles = trimesh_obj.faces

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return o3d_mesh


def pyvista2trimesh(pyvista_polydata):
    pyvista_polydata = pyvista_polydata.triangulate()
    return trimesh.Trimesh(pyvista_polydata.points, pyvista_polydata.faces.reshape(-1, 4)[:, 1:], process=False)


def trimesh2pyvista(trimesh_obj):
    return pv.make_tri_mesh(trimesh_obj.vertices, trimesh_obj.faces)


def pyvista2o3d(pyvista_polydata):
    return trimesh2o3d(pyvista2trimesh(pyvista_polydata))


def o3d2pyvista(o3d_mesh):
    return trimesh2pyvista(o3d2trimesh(o3d_mesh))


def visualize_geometries(geometries: list):
    o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True, mesh_show_back_face=True)


def pcd2tensor(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points, dtype=np.float32)
    return torch.from_numpy(points)


def np2tensor(array: np.ndarray):
    return torch.from_numpy(array)