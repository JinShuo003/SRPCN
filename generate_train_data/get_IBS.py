"""
计算IBS
"""
import logging
import multiprocessing
import os
import re

import open3d as o3d

from utils import geometry_utils, path_utils, ibs_utils, log_utils


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
        subdivide_max_edge = self.specs.get("caculate_options").get("subdivide_max_edge")
        sample_num = self.specs.get("caculate_options").get("sample_num")
        sample_method = self.specs.get("caculate_options").get("sample_method")
        clip_border_type = self.specs.get("caculate_options").get("clip_border_options").get("clip_border_type")
        clip_sphere_radius = self.specs.get("caculate_options").get("clip_border_options").get("clip_sphere_radius")
        clip_border_magnification = self.specs.get("caculate_options").get("clip_border_options").get("clip_border_magnification")
        max_iterate_time = self.specs.get("caculate_options").get("max_iterate_time")
        show_iterate_result = self.specs.get("caculate_options").get("show_iterate_result")
        max_resample_points = self.specs.get("caculate_options").get("max_resample_points")
        max_points_for_compute = self.specs.get("caculate_options").get("max_points_for_compute")
        simplify = self.specs.get("caculate_options").get("simplify")
        max_triangle_num = self.specs.get("caculate_options").get("max_triangle_num")

        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])

        ibs = ibs_utils.IBS(geometry_utils.o3d2trimesh(mesh1),
                            geometry_utils.o3d2trimesh(mesh2),
                            subdivide_max_edge=subdivide_max_edge,
                            sample_method=sample_method,
                            sample_num=sample_num,
                            clip_border_type=clip_border_type,
                            clip_sphere_radius=clip_sphere_radius,
                            clip_border_magnification=clip_border_magnification,
                            max_iterate_time=max_iterate_time,
                            show_iterate_result=show_iterate_result,
                            max_resample_points=max_resample_points,
                            max_points_for_compute=max_points_for_compute,
                            simplify=simplify,
                            max_triangle_num=max_triangle_num,
                            logger=self.logger)
        ibs.launch()
        ibs_o3d = ibs.get_ibs_o3d()

        return ibs_o3d

    def handle_scene(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        ibs_mesh_o3d = self.get_ibs_mesh_o3d(geometries_path)
        save_ibs_mesh(self.specs, scene, ibs_mesh_o3d)


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
    config_filepath = 'configs/get_IBS.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("ibs_mesh_save_dir"))

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
