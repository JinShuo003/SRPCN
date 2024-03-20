from pathlib import Path
import numpy as np
import re
import trimesh
import open3d as o3d
import os

from utils.geometry_utils import get_sphere_mesh, o3d2trimesh, trimesh2o3d


if __name__ == '__main__':
    base_path = Path(os.getcwd())
    scene_name = 'scene1.1000'
    data_path = base_path / scene_name
    filename_list = os.listdir(data_path.as_posix())
    filename_list = [filename for filename in filename_list if filename.endswith('.obj')]

    mesh_list = []
    for filename in filename_list:
        filepath = (data_path / filename).as_posix()
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh = o3d2trimesh(mesh)
        mesh_list.append(mesh)

    mesh = trimesh.boolean.union(mesh_list, engine='blender')
    mesh = trimesh2o3d(mesh)
    # mesh.remove_unreferenced_vertices()
    # mesh.remove_duplicated_vertices()
    # mesh.remove_degenerate_triangles()

    o3d.io.write_triangle_mesh(filename=(data_path / "{}.obj".format('union_result')).as_posix(), mesh=mesh)
