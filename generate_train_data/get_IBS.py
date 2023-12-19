"""
计算成对Mesh的IBS面
"""
import logging
import multiprocessing
import os
import re
import numpy as np

import open3d as o3d
import pyibs

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


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger

    def get_ibs_mesh_o3d(self, geometries_path: dict):
        subdevide_max_edge = specs.get("caculate_options").get("subdevide_max_edge")
        sample_num = specs.get("caculate_options").get("sample_num")
        sample_method = specs.get("caculate_options").get("sample_method")
        clip_border_ratio = specs.get("caculate_options").get("clip_border_ratio")

        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])
        mesh1.paint_uniform_color((1, 0, 0))
        mesh2.paint_uniform_color((0, 1, 0))

        pcd1 = geometry_utils.read_point_cloud(geometries_path["pcd1"])
        pcd2 = geometry_utils.read_point_cloud(geometries_path["pcd2"])

        ibs = pyibs.IBS(np.asarray(pcd1.points), np.asarray(pcd2.points))
        ibs_o3d = geometry_utils.trimesh2o3d(ibs.mesh)
        ibs_o3d.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([ibs_o3d, mesh1, mesh2], mesh_show_wireframe=True, mesh_show_back_face=True)

        ibs = ibs_utils.IBS(geometry_utils.o3d2trimesh(mesh1),
                            geometry_utils.o3d2trimesh(mesh2),
                            subdevide_max_edge=subdevide_max_edge,
                            sample_num=sample_num,
                            sample_method=sample_method,
                            clip_border_ratio=clip_border_ratio,
                            logger=self.logger)
        ibs_o3d = geometry_utils.trimesh2o3d(ibs.ibs)
        ibs_o3d.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([ibs_o3d, mesh1, mesh2], mesh_show_wireframe=True, mesh_show_back_face=True)

        return ibs_o3d

    def handle_scene(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        ibs_mesh_o3d = self.get_ibs_mesh_o3d(geometries_path)

        save_ibs_mesh(self.specs, scene, ibs_mesh_o3d)


def get_logger(scene: str):
    _logger = logging.getLogger()
    _logger.setLevel("INFO")
    log_path = "logs/get_IBS/{}.log"
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
    config_filepath = 'configs/get_IBS.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("ibs_mesh_save_dir"))

    logger = logging.getLogger("get_IBS")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    pool = multiprocessing.Pool(processes=specs.get("processNum"))

    # 参数
    view_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                view_list.append(filename)

    # for filename in view_list:
    #     logger.info("current scene: {}".format(filename))
    #     pool.apply_async(my_process, (filename, specs))
    #
    # # 关闭进程池
    # pool.close()
    # pool.join()

    for filename in view_list:
        logger.info("current scene: {}".format(filename))
        _logger, file_handler, stream_handler = get_logger(filename)

        trainDataGenerator = TrainDataGenerator(specs, _logger)
        trainDataGenerator.handle_scene(filename)

        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)

