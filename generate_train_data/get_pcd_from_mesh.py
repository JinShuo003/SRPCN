"""
从Mesh上采集点云
"""
import logging
import multiprocessing
import os
import re
import numpy as np

import open3d as o3d
from utils import path_utils, log_utils


def save_obj_pcd(specs, scene, pcd: tuple):
    pcd_save_dir = specs.get("path_options").get("pcd_save_dir")
    category = re.match(specs.get("path_options").get("format_info").get("category_re"), scene).group()
    if not os.path.isdir(os.path.join(pcd_save_dir, category)):
        os.makedirs(os.path.join(pcd_save_dir, category))

    pcd1_filename = '{}_0.ply'.format(scene)
    pcd1_path = os.path.join(pcd_save_dir, category, pcd1_filename)
    pcd2_filename = '{}_1.ply'.format(scene)
    pcd2_path = os.path.join(pcd_save_dir, category, pcd2_filename)

    o3d.io.write_point_cloud(pcd1_path, pcd[0])
    o3d.io.write_point_cloud(pcd2_path, pcd[1])


def save_ibs_pcd(specs, scene, pcd):
    pcd_save_dir = specs.get("path_options").get("pcd_save_dir")
    category = re.match(specs.get("path_options").get("format_info").get("category_re"), scene).group()
    if not os.path.isdir(os.path.join(pcd_save_dir, category)):
        os.makedirs(os.path.join(pcd_save_dir, category))

    ibs_filename = '{}.ply'.format(scene)
    ibs_path = os.path.join(pcd_save_dir, category, ibs_filename)

    o3d.io.write_point_cloud(ibs_path, pcd)


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger

    def get_obj_pcd(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)

        sample_num = self.specs["sample_num"]
        mesh1 = o3d.io.read_triangle_mesh(geometries_path["mesh1"])
        mesh2 = o3d.io.read_triangle_mesh(geometries_path["mesh2"])
        self.logger.info("mesh1 with {} vertices, {} triangles".
                         format(np.asarray(mesh1.vertices).shape[0], np.asarray(mesh1.triangles).shape[0]))
        self.logger.info("mesh2 with {} vertices, {} triangles".
                         format(np.asarray(mesh2.vertices).shape[0], np.asarray(mesh2.triangles).shape[0]))

        pcd1 = mesh1.sample_points_poisson_disk(sample_num)
        pcd2 = mesh2.sample_points_poisson_disk(sample_num)
        self.logger.info("get {} points from mesh1".format(np.asarray(pcd1.points).shape[0]))
        self.logger.info("get {} points from mesh2".format(np.asarray(pcd2.points).shape[0]))

        save_obj_pcd(self.specs, scene, (pcd1, pcd2))

    def get_ibs_pcd(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        sample_num = self.specs["sample_num"]

        ibs = o3d.io.read_triangle_mesh(geometries_path["ibs"])
        self.logger.info("ibs with {} vertices, {} triangles".
                         format(np.asarray(ibs.vertices).shape[0], np.asarray(ibs.triangles).shape[0]))

        pcd = ibs.sample_points_poisson_disk(sample_num)
        self.logger.info("get {} points from ibs".format(np.asarray(pcd.points).shape[0]))

        save_ibs_pcd(self.specs, scene, pcd)

    def handle_scene(self, scene):
        self.logger.info("current type: {}".format(self.specs.get("type")))
        if self.specs.get("type") == "object":
            self.get_obj_pcd(scene)
        elif self.specs.get("type") == "ibs":
            self.get_ibs_pcd(scene)


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
    config_filepath = 'configs/get_pcd_from_mesh.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("pcd_save_dir"))

    logger = logging.getLogger("get_pcd_from_mesh")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.WARNING)
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
            logger.warning("current scene: {}".format(filename))
            _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"),
                                                                         filename)

            trainDataGenerator = TrainDataGenerator(specs, _logger)
            trainDataGenerator.handle_scene(filename)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)
