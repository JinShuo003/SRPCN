"""
对原始Mesh进行碰撞检测
"""
import logging
import multiprocessing
import os
import re

import numpy as np
import open3d as o3d
import trimesh.collision

from utils import geometry_utils, path_utils, ibs_utils


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

    def handle_scene(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])
        mesh1.paint_uniform_color((0, 0, 1))
        mesh2.paint_uniform_color((0, 1, 0))

        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object('obj1', geometry_utils.o3d2trimesh(mesh1))
        collision_manager.add_object('obj2', geometry_utils.o3d2trimesh(mesh2))
        is_collision, data = collision_manager.in_collision_internal(return_data=True)
        if is_collision:
            self.logger.info("collision occured, amount: {}".format(len(data)))
            collision_points = []
            for collision_data in data:
                collision_points.append(collision_data.point)
            collision_pcd = geometry_utils.get_pcd_from_np(np.array(collision_points))
            aabb = collision_pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            o3d.visualization.draw_geometries([aabb, mesh1, mesh2], mesh_show_wireframe=True, mesh_show_back_face=True)

        else:
            self.logger.info("No collision")


def get_logger(scene: str):
    _logger = logging.getLogger()
    _logger.setLevel("INFO")
    log_path = "logs/collision_test/{}.log"
    path_utils.generate_path(os.path.split(log_path)[0])

    file_handler = logging.FileHandler(log_path.format(scene), mode="w")
    file_handler.setLevel(level=logging.INFO)
    _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    _logger.addHandler(stream_handler)

    return _logger, file_handler, stream_handler


def my_process(scene, specs):
    _logger, file_handler, stream_handler = get_logger(scene)
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

    pool = multiprocessing.Pool(processes=specs.get("processNum"))
    for filename in view_list:
        logger.info("current scene: {}".format(filename))
        pool.apply_async(my_process, (filename, specs))
    pool.close()
    pool.join()

    # for filename in view_list:
    #     logger.info("current scene: {}".format(filename))
    #     _logger, file_handler, stream_handler = get_logger(filename)
    #
    #     collision_analyzer = CollisionAnalyzer(specs, _logger)
    #     collision_analyzer.handle_scene(filename)
    #
    #     _logger.removeHandler(file_handler)
    #     _logger.removeHandler(stream_handler)

