"""
多进程计算点云的ibs面
"""
import multiprocessing
import os
import re
import time
import open3d as o3d
import numpy as np
from ordered_set import OrderedSet
import json
from utils.calIBS import IBS
from utils.geometry_adaptor import *


def parseConfig(config_filepath: str = './config/generatePointCloud.json'):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def getFilenameTree(specs: dict, base_path: str):
    """以base_path为基准构建文件树，文件树的格式为
    {
    'scene1': {
              'scene1.1000' :[scene1.1000_view0_0,
                              scene1.1000_view0_1],
              'scene1.1001' :[scene1.1001_view0_0,
                              scene1.1001_view0_1]
              }
    }
    """
    # 构建文件树
    base_path = specs[base_path]

    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    filename_re = specs["filename_re"]

    handle_category = specs["handle_category"]
    handle_scene = specs["handle_scene"]
    handle_filename = specs["handle_filename"]

    filename_tree = dict()
    folder_info = os.walk(base_path)
    for dir_path, dir_names, filenames in folder_info:
        # 顶级目录不做处理
        if dir_path == base_path:
            continue
        category = dir_path.split('\\')[-1]
        if not regular_match(handle_category, category):
            continue
        if category not in filename_tree:
            filename_tree[category] = dict()
        for filename in filenames:
            scene = re.match(scene_re, filename).group()
            if not regular_match(handle_scene, scene):
                continue
            if scene not in filename_tree[category]:
                filename_tree[category][scene] = OrderedSet()
            filename = re.match(filename_re, filename).group()
            if not regular_match(handle_filename, filename):
                continue
            filename_tree[category][scene].add(filename)
    # 将有序集合转为列表
    filename_tree_copy = dict()
    for category in filename_tree.keys():
        for scene in filename_tree[category]:
            filename_tree[category][scene] = list(filename_tree[category][scene])
            if len(filename_tree[category][scene]) != 0:
                if category not in filename_tree_copy.keys():
                    filename_tree_copy[category] = dict()
                filename_tree_copy[category][scene] = filename_tree[category][scene]
    return filename_tree_copy


def generatePath(specs: dict, path_list: list):
    """
    检查specs中的path是否存在，不存在则创建
    """
    for path in path_list:
        if not os.path.isdir(specs[path]):
            os.makedirs(specs[path])


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)


def getGeometriesPath(specs, filename):
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    filename_re = specs["filename_re"]

    category = re.match(category_re, filename).group()
    scene = re.match(scene_re, filename).group()
    filename = re.match(filename_re, filename).group()

    geometries_path = dict()

    pcd_dir = specs["pcd_dir"]

    pcd1_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_filename = '{}_{}.ply'.format(filename, 1)

    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)

    return geometries_path


def save_ibs_mesh(specs, scene, ibs_mesh_o3d):
    mesh_dir = specs['ibs_mesh_save_dir']
    category = re.match(specs['category_re'], scene).group()
    # mesh_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    ibs_mesh_filename = '{}.obj'.format(scene)
    mesh_path = os.path.join(mesh_dir, category, ibs_mesh_filename)
    o3d.io.write_triangle_mesh(mesh_path, ibs_mesh_o3d)


# ----------------------------------------其他-------------------------------------------

class GeometryHandler:
    def __init__(self):
        pass

    def get_pcd_normalize_para(self, pcd):
        pcd_np = np.asarray(pcd.points)
        # 求点云的中心
        centroid = np.mean(pcd_np, axis=0)
        # 求长轴长度
        scale = np.max(np.sqrt(np.sum(pcd_np ** 2, axis=1)))
        return centroid, scale

    def geometry_transform(self, geometry, centroid, scale):
        coor = self.get_unit_coordinate()
        geometry.translate(-centroid)
        geometry.scale(1 / scale, np.array([0, 0, 0]))

    def get_unit_sphere(self):
        # 创建单位球点云
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.compute_vertex_normals()
        return mesh_sphere

    def get_unit_sphere_pcd(self):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        pcd = mesh_sphere.sample_points_uniformly(256)
        return pcd

    def get_unit_coordinate(self):
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        coord_frame.compute_vertex_normals()
        return coord_frame


class TrainDataGenerator:
    def __init__(self, specs):
        self.geometryHandler = GeometryHandler()
        self.specs = specs
        self.geometries_path = None
        # pcd
        self.pcd1 = None
        self.pcd2 = None
        self.ibs_mesh = None

    def get_pcd(self):
        """读取残缺点云文件"""
        self.pcd1 = o3d.io.read_point_cloud(self.geometries_path["pcd1"])
        self.pcd1.paint_uniform_color((1, 0, 0))
        self.pcd2 = o3d.io.read_point_cloud(self.geometries_path["pcd2"])
        self.pcd2.paint_uniform_color((0, 1, 0))

    def get_init_geometries(self):
        """获取初始几何体"""
        self.get_pcd()

    def combine_pcds(self, pcd1, pcd2):
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        combined_points = np.concatenate((points1, points2))
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        return combined_pcd

    def combine_meshes(self, mesh1, mesh2):
        # 获取第一个Mesh的顶点和面数据
        vertices1 = mesh1.vertices
        faces1 = np.asarray(mesh1.triangles)

        # 获取第二个Mesh的顶点和面数据
        vertices2 = mesh2.vertices
        faces2 = np.asarray(mesh2.triangles)

        # 将第二个Mesh的顶点坐标添加到第一个Mesh的顶点列表中
        combined_vertices = np.concatenate((vertices1, vertices2))

        # 更新第二个Mesh的面索引，使其适应顶点索引的变化
        faces2 += len(vertices1)

        # 将两个Mesh的面数据合并
        combined_faces = np.concatenate((faces1, faces2))

        # 创建一个新的Mesh对象
        combined_mesh = o3d.geometry.TriangleMesh()

        # 设置新的Mesh的顶点和面数据
        combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
        combined_mesh.triangles = o3d.utility.Vector3iVector(combined_faces)

        return combined_mesh

    def o3d2trimesh(self, o3d_mesh):
        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        return tri_mesh

    def trimesh2o3d(self, tri_mesh):
        vertices = tri_mesh.vertices
        triangles = tri_mesh.faces

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        return o3d_mesh

    def aabb2pcd(self, aabb):
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()

        # 构建八个顶点的坐标
        vertices = [
            [min_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]]
        ]

        # 将顶点坐标放入NumPy数组
        vertices_np = np.array(vertices, dtype=np.float32)
        return vertices_np

    def get_ibs_mesh_o3d(self):
        points0 = np.asarray(self.pcd1.points)
        points1 = np.asarray(self.pcd2.points)
        ibs = IBS(points0, points1, n=2048, bounded=True)
        ibs_mesh_o3d = trimesh2o3d(ibs.mesh)
        self.ibs_mesh = ibs_mesh_o3d
        return ibs_mesh_o3d

    def handle_scene(self, scene):
        """处理当前场景，包括采集多角度的残缺点云、计算直接法和间接法网络的sdf gt、计算残缺点云下的ibs"""
        # ------------------------------获取点云数据，包括完整点云和各个视角的残缺点云--------------------------
        self.geometries_path = getGeometriesPath(self.specs, scene)
        self.get_init_geometries()
        ibs_mesh_o3d = self.get_ibs_mesh_o3d()

        # sphere = self.geometryHandler.get_unit_sphere_pcd()
        # o3d.visualization.draw_geometries([ibs_mesh_o3d, self.pcd1, self.pcd2, sphere], mesh_show_wireframe=True, mesh_show_back_face=True)
        save_ibs_mesh(self.specs, scene, ibs_mesh_o3d)


def my_process(scene, specs):
    process_name = multiprocessing.current_process().name
    print(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs)
    try:
        trainDataGenerator.handle_scene(scene)
        print(f"scene: {scene} succeed")
    except:
        print(f"scene: {scene} failed")


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'config/getIBS.json'
    specs = parseConfig(config_filepath)
    # 构建文件树
    filename_tree = getFilenameTree(specs, "pcd_dir")
    # 处理文件夹，不存在则创建
    generatePath(specs, ["ibs_mesh_save_dir"])

    # 创建进程池，指定进程数量
    pool = multiprocessing.Pool(processes=8)
    # 参数
    view_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                view_list.append(filename)
    # 使用进程池执行任务，返回结果列表
    for filename in view_list:
        category_num = int(filename[5])-1
        pool.apply_async(my_process, (filename, specs))

    # 关闭进程池
    pool.close()
    pool.join()

