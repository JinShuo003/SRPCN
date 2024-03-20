import os
from pathlib import Path
import numpy as np
import re
import trimesh
import open3d as o3d

from utils.geometry_utils import get_sphere_mesh, o3d2trimesh, trimesh2o3d


def get_mas_data(mat_filepath):
    data = np.load(mat_filepath)
    center = data["center"]
    radius = data["radius"]
    return center, radius


def get_union_mais(mat_file, step, idx):
    center, radius = get_mas_data(mat_file)
    sphere_num = radius.shape[0]

    mesh_list = []
    begin = step*idx
    for i in range(begin, begin+step):
        center_ = center[i]
        radius_ = radius[i]
        mesh = get_sphere_mesh(center=center_, radius=radius_)
        mesh = o3d2trimesh(mesh)
        mesh_list.append(mesh)

    mesh = trimesh.boolean.union(mesh_list, engine='blender')
    return mesh


if __name__ == '__main__':
    base_path = "D:\\dataset\\IBPCDC"
    filename = "scene1.1000"
    category_patten = "scene\\d"
    category = re.match(category_patten, filename).group()
    base_path = Path(base_path)
    mat_path = base_path / 'MedialAxis' / 'INTE' / category
    cwd = Path(os.getcwd())
    output_path = cwd / filename
    if not os.path.exists(output_path.as_posix()):
        os.makedirs(output_path.as_posix())

    mat_filename = '{}'.format(filename)
    mat_file = mat_path / '{}.npz'.format(mat_filename)

    sphere_num_total = 2048
    group_num = 16
    step = sphere_num_total // group_num
    for idx in range(group_num):
        mesh = get_union_mais(mat_file, step, idx)
        mesh = trimesh2o3d(mesh)
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()

        filepath = output_path / "{}_{}.obj".format(filename, idx)
        o3d.io.write_triangle_mesh(filename=filepath.as_posix(), mesh=mesh)
        print('saved a group of sphere, idx: {}'.format(idx))
