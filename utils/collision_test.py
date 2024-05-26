"""
碰撞检测，检查训练数据有无问题
"""
import logging
import multiprocessing
import os
import re

import open3d as o3d
import trimesh.collision

from utils import geometry_utils, path_utils, log_utils


def save_ibs_mesh(specs, scene, ibs_mesh_o3d):
    mesh_dir = specs.get("path_options").get("ibs_mesh_save_dir")
    category = re.match(specs.get("path_options").get("format_info").get("category_re"), scene).group()
    # mesh_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    ibs_mesh_filename = '{}.obj'.format(scene)
    mesh_path = os.path.join(mesh_dir, category, ibs_mesh_filename)
    o3d.io.write_triangle_mesh(mesh_path, ibs_mesh_o3d)


class CollisionAnalyzer:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger

    def collision_test_origin_mesh(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])

        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object('obj1', geometry_utils.o3d2trimesh(mesh1))
        collision_manager.add_object('obj2', geometry_utils.o3d2trimesh(mesh2))
        is_collision, data = collision_manager.in_collision_internal(return_data=True)
        if is_collision:
            self.logger.info("collision occured, amount: {}".format(len(data)))
        else:
            self.logger.info("No collision")

    def collision_test_ibs(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])
        ibs = geometry_utils.read_mesh(geometries_path["ibs"])

        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object('obj1', geometry_utils.o3d2trimesh(mesh1))
        collision_manager.add_object('obj2', geometry_utils.o3d2trimesh(mesh2))
        is_collision, data = collision_manager.in_collision_single(geometry_utils.o3d2trimesh(ibs), return_data=True)
        if is_collision:
            self.logger.info("collision occured, amount: {}".format(len(data)))
        else:
            self.logger.info("No collision")

    def handle_scene(self, scene):
        self.logger.info("collision test type: {}".format(self.specs.get("type")))
        if self.specs.get("type") == "origin_mesh":
            self.collision_test_origin_mesh(scene)
        elif self.specs.get("type") == "ibs":
            self.collision_test_ibs(scene)


def my_process(scene, specs):
    _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)
    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    collision_analyzer = CollisionAnalyzer(specs, _logger)

    try:
        collision_analyzer.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e.message))
    finally:
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)


if __name__ == '__main__':
    config_filepath = 'configs/collision_test.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))

    logger = logging.getLogger("collision_test")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

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

            trainDataGenerator = CollisionAnalyzer(specs, _logger)
            trainDataGenerator.handle_scene(filename)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)

