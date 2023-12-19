import open3d as o3d
import numpy as np
import open3d.cpu.pybind.geometry
import trimesh
import pyvista as pv


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
        pcd_o3d = open3d.geometry.PointCloud()
        pcd_o3d.points = open3d.utility.Vector3dVector(pcd)
    pcd_np = np.asarray(pcd_o3d.points)
    axis_aligned_bounding_box = pcd_o3d.get_axis_aligned_bounding_box()
    centroid = np.asarray(axis_aligned_bounding_box.get_center())
    scale = np.max(np.sqrt(np.sum((pcd_np-centroid) ** 2, axis=1)))
    return centroid, scale


def geometry_transform(geometry, centroid, scale):
    """
    按照归一化参数将open3d.Geometry执行平移和缩放变换
    Args:
        geometry: open3d.Geometry
        centroid: Geometry的轴对齐包围盒中心点
        scale: Geometry的尺度
    Returns:
        归一化后的open3d.geometry，几何中心位于(0, 0, 0)，所有几何元素位于单位球内
    """
    geometry.translate(-centroid)
    geometry.scale(1 / scale, np.array([0, 0, 0]))
    return geometry


def get_sphere_mesh(radius=1.0):
    """
    获取半径为radius，格式为open3d.TriangleMesh的球
    Args:
        radius: 球半径
    Returns:
        r=radius，format=open3d.TriangleMesh的球
    """
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_mesh.compute_vertex_normals()
    return sphere_mesh


def get_sphere_pcd(center=(0, 0, 0), radius=1, points_num=256):
    """
    获取r=radius，格式为open3d.PointCloud的球
    Args:
        center: 球心
        radius: 球半径
        points_num: 点云点数
    Returns:
        r=radius，点数=points_num，format=open3d.PointCloud的球
    """
    sphere_mesh = pyvista2o3d(pv.Sphere(radius, center))
    sphere_pcd = sphere_mesh.sample_points_uniformly(points_num)
    return sphere_pcd


def get_unit_coordinate(size=1):
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
