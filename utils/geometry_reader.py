import open3d as o3d


def read_point_cloud(path):
    return o3d.io.read_point_cloud(path)


def read_mesh(path):
    return o3d.io.read_triangle_mesh(path)
