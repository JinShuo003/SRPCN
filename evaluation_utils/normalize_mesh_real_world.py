"""
将真实场景中的若干个Mesh共同归一化到单位球内
"""
from copy import deepcopy
import numpy as np
import open3d as o3d

from utils import geometry_utils


def read_mesh(mesh_filename_list: list):
    mesh_list = []
    for filename in mesh_filename_list:
        mesh = geometry_utils.read_mesh(filename)
        mesh_list.append(mesh)
    return mesh_list


def combine_mesh(mesh_list: list):
    vertices_total = np.zeros((0, 3))
    faces_total = np.zeros((0, 3))
    for mesh in mesh_list:
        mesh_copy = deepcopy(mesh)
        vertices = mesh_copy.vertices
        faces = np.asarray(mesh_copy.triangles)
        faces += len(vertices_total)
        vertices_total = np.concatenate((vertices_total, vertices))
        faces_total = np.concatenate((faces_total, faces))

    mesh_total = o3d.geometry.TriangleMesh()
    mesh_total.vertices = o3d.utility.Vector3dVector(vertices_total)
    mesh_total.triangles = o3d.utility.Vector3iVector(faces_total)

    return mesh_total


def get_normalize_para(mesh):
    aabb = mesh.get_axis_aligned_bounding_box()
    centroid = aabb.get_center()
    max_bound = aabb.get_max_bound()
    min_bound = aabb.get_min_bound()
    scale = np.linalg.norm(max_bound - min_bound)
    return centroid, scale/2


if __name__ == '__main__':
    mesh_filename_list = [
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene1\scene1.1012_0.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene1\scene1.1012_1.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene2\scene2.1000_0.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene2\scene2.1000_1.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene3\scene3.1007_0.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene3\scene3.1007_1.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene9\scene9.1001_0.obj',
        r'D:\dataset\IBSNet\evaluateData\real-world\mesh\scene9\scene9.1001_1.obj'
    ]
    normalize_radius = 0.5

    # read mesh
    mesh_list = read_mesh(mesh_filename_list)
    # o3d.visualization.draw_geometries(mesh_list)

    # get combined mesh
    mesh_total = combine_mesh(mesh_list)
    # o3d.visualization.draw_geometries([mesh_total])

    # get total normalize parameters
    centroid, scale = get_normalize_para(mesh_total)
    print(centroid)
    print(scale)

    sphere = geometry_utils.get_sphere_pcd(radius=0.5)

    # normalize every mesh
    normalized_mesh_list = []
    for mesh in mesh_list:
        mesh = geometry_utils.normalize_geometry(mesh, centroid, scale)
        mesh.scale(normalize_radius, np.array([0, 0, 0]))
        normalized_mesh_list.append(mesh)
    # o3d.visualization.draw_geometries(normalized_mesh_list + [sphere])

    # save every mesh
    for i in range(len(normalized_mesh_list)):
        o3d.io.write_triangle_mesh(mesh_filename_list[i], normalized_mesh_list[i])
