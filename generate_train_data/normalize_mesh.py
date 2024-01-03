"""
归一化Mesh
"""
import copy
import logging
import multiprocessing
import os
import re

import numpy as np
import open3d as o3d

from utils import geometry_utils, path_utils, log_utils


def getGeometriesPath(specs, scene):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    category = re.match(category_re, scene).group()
    scene = re.match(scene_re, scene).group()

    geometries_path = dict()

    mesh_dir = specs.get("path_options").get("geometries_dir").get("mesh_dir")
    IOUgt_dir = specs.get("path_options").get("geometries_dir").get("IOUgt_dir")

    mesh1_filename = '{}_{}.off'.format(scene, 0)
    mesh2_filename = '{}_{}.off'.format(scene, 1)
    IOUgt_filename = '{}.txt'.format(scene)

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometries_path['IOUgt'] = os.path.join(IOUgt_dir, category, IOUgt_filename)

    return geometries_path


def save_mesh(specs, scene, mesh1, mesh2):
    mesh_dir = specs.get("path_options").get("mesh_normalize_save_dir")
    category = re.match(specs.get("path_options").get("format_info").get("category_re"), scene).group()
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    mesh1_filename = '{}_0.obj'.format(scene)
    mesh2_filename = '{}_1.obj'.format(scene)
    mesh1_path = os.path.join(mesh_dir, category, mesh1_filename)
    mesh2_path = os.path.join(mesh_dir, category, mesh2_filename)

    o3d.io.write_triangle_mesh(mesh1_path, mesh1)
    o3d.io.write_triangle_mesh(mesh2_path, mesh2)


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger
        self.geometries_path = None

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

    def get_normalize_para(self, mesh):
        aabb = mesh.get_axis_aligned_bounding_box()
        centroid = aabb.get_center()
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        scale = np.linalg.norm(max_bound - min_bound)
        return centroid, scale/2

    def get_IOU(self):
        """获取交互区域的aabb框"""
        with open(self.geometries_path["IOUgt"], 'r') as file:
            data = file.readlines()
            line1 = data[0].strip('\n').strip(' ').split(' ')
            line2 = data[1].strip('\n').strip(' ').split(' ')
            min_bound = np.array([float(item) for item in line1])
            max_bound = np.array([float(item) for item in line2])
            return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    def handle_scene(self, scene):
        """读取mesh，组合后求取归一化参数，然后分别归一化到单位球内，保存结果"""
        normalize_radius = self.specs.get("normalize_radius")
        self.geometries_path = path_utils.get_geometries_path(self.specs, scene)

        mesh1 = o3d.io.read_triangle_mesh(self.geometries_path["mesh1"])
        mesh2 = o3d.io.read_triangle_mesh(self.geometries_path["mesh2"])
        combined_mesh = self.combine_meshes(copy.deepcopy(mesh1), copy.deepcopy(mesh2))
        centroid, scale = self.get_normalize_para(combined_mesh)
        print(centroid)
        print(scale)

        mesh1 = geometry_utils.geometry_transform(mesh1, centroid, scale)
        mesh2 = geometry_utils.geometry_transform(mesh2, centroid, scale)
        mesh1.scale(normalize_radius, np.array([0, 0, 0]))
        mesh2.scale(normalize_radius, np.array([0, 0, 0]))

        save_mesh(self.specs, scene, mesh1, mesh2)


def my_process(scene, specs):
    _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)
    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs, _logger)

    try:
        trainDataGenerator.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e.message))
    finally:
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)


if __name__ == '__main__':
    config_filepath = 'configs/normalize_mesh.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("mesh_normalize_save_dir"))

    logger = logging.getLogger("get_IBS")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # 参数
    view_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                view_list.append(filename)

    if specs.get("use_process_pool"):
        pool = multiprocessing.Pool(processes=specs.get("process_num"))

        for filename in view_list:
            logger.info("current scene: {}".format(filename))
            pool.apply_async(my_process, (filename, specs))

        pool.close()
        pool.join()
    else:
        for filename in view_list:
            logger.info("current scene: {}".format(filename))
            _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), filename)

            trainDataGenerator = TrainDataGenerator(specs, _logger)
            trainDataGenerator.handle_scene(filename)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)
