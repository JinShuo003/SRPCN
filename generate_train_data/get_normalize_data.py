"""
将成对的完整点云、残缺点云、IBS各自进行归一化，并保存归一化参数
"""
import logging
import multiprocessing
import os
import re

import open3d as o3d
import numpy as np

from utils import path_utils, geometry_utils, log_utils


def get_geometry_path(specs, data_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, data_name).group()
    scene = re.match(scene_re, data_name).group()
    filename = re.match(filename_re, data_name).group()

    geometries_path = dict()

    pcd_complete_dir = specs.get("path_options").get("geometries_dir").get("pcd_complete_dir")
    pcd_partial_dir = specs.get("path_options").get("geometries_dir").get("pcd_partial_dir")
    IBS_dir = specs.get("path_options").get("geometries_dir").get("IBS_dir")
    Medial_axis_sphere_dir = specs.get("path_options").get("geometries_dir").get("Medial_axis_sphere_dir")

    pcd1_complete_filename = '{}_{}.ply'.format(scene, 0)
    pcd2_complete_filename = '{}_{}.ply'.format(scene, 1)
    pcd1_partial_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_partial_filename = '{}_{}.ply'.format(filename, 1)
    IBS_filename = '{}.ply'.format(scene)
    Medial_axis_sphere_filename = '{}.npz'.format(scene)

    geometries_path['pcd1_complete'] = os.path.join(pcd_complete_dir, category, pcd1_complete_filename)
    geometries_path['pcd2_complete'] = os.path.join(pcd_complete_dir, category, pcd2_complete_filename)
    geometries_path['pcd1_partial'] = os.path.join(pcd_partial_dir, category, pcd1_partial_filename)
    geometries_path['pcd2_partial'] = os.path.join(pcd_partial_dir, category, pcd2_partial_filename)
    geometries_path['IBS'] = os.path.join(IBS_dir, category, IBS_filename)
    geometries_path['Medial_axis_sphere'] = os.path.join(Medial_axis_sphere_dir, category, Medial_axis_sphere_filename)

    return geometries_path


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger
        self.geometries_path = None

    def get_pcd_complete_save_path(self, data_name, tag):
        pcd_complete_save_dir = self.specs.get("path_options").get("pcd_complete_save_dir")

        category_re = self.specs.get("path_options").get("format_info").get("category_re")
        scene_re = self.specs.get("path_options").get("format_info").get("scene_re")

        category = re.match(category_re, data_name).group()
        scene = re.match(scene_re, data_name).group()

        save_path = os.path.join(pcd_complete_save_dir, category, "{}_{}.ply".format(scene, tag))
        return save_path

    def get_pcd_partial_save_path(self, data_name, tag):
        pcd_partial_save_dir = self.specs.get("path_options").get("pcd_partial_save_dir")

        category_re = self.specs.get("path_options").get("format_info").get("category_re")
        filename_re = self.specs.get("path_options").get("format_info").get("filename_re")

        category = re.match(category_re, data_name).group()
        filename = re.match(filename_re, data_name).group()

        save_path = os.path.join(pcd_partial_save_dir, category, "{}_{}.ply".format(filename, tag))
        return save_path

    def get_IBS_save_path(self, data_name, tag):
        IBS_save_dir = self.specs.get("path_options").get("IBS_save_dir")

        category_re = self.specs.get("path_options").get("format_info").get("category_re")
        scene_re = self.specs.get("path_options").get("format_info").get("scene_re")

        category = re.match(category_re, data_name).group()
        scene = re.match(scene_re, data_name).group()

        save_path = os.path.join(IBS_save_dir, category, "{}_{}.ply".format(scene, tag))
        return save_path

    def get_medial_axis_sphere__save_path(self, data_name, tag):
        Medial_axis_sphere_save_dir = self.specs.get("path_options").get("Medial_axis_save_dir")

        category_re = self.specs.get("path_options").get("format_info").get("category_re")
        scene_re = self.specs.get("path_options").get("format_info").get("scene_re")

        category = re.match(category_re, data_name).group()
        scene = re.match(scene_re, data_name).group()

        save_path = os.path.join(Medial_axis_sphere_save_dir, category, "{}_{}.npz".format(scene, tag))
        return save_path

    def get_normalize_para_save_path(self, data_name, tag):
        normalize_para_save_dir = self.specs.get("path_options").get("normalize_para_save_dir")

        category_re = self.specs.get("path_options").get("format_info").get("category_re")
        scene_re = self.specs.get("path_options").get("format_info").get("scene_re")

        category = re.match(category_re, data_name).group()
        scene = re.match(scene_re, data_name).group()

        save_path = os.path.join(normalize_para_save_dir, category, "{}_{}.txt".format(scene, tag))
        return save_path

    def save_pcd(self, pcd, save_path):
        save_dir, filename = os.path.split(save_path)
        path_utils.generate_path(save_dir)

        o3d.io.write_point_cloud(save_path, pcd)

    def save_medial_axis_sphere(self, center, radius, direction, save_path):
        save_dir, filename = os.path.split(save_path)
        path_utils.generate_path(save_dir)

        np.savez(save_path, center=center, radius=radius, direction=direction)

    def save_normalize_para(self, translate, scale, save_path):
        save_dir, filename = os.path.split(save_path)
        path_utils.generate_path(save_dir)

        data = []
        data += translate.tolist()
        data.append(scale)
        with open(save_path, 'w') as f:
            f.write(",".join(map(str, data)))

    def handle_scene(self, scene):
        self.geometries_path = get_geometry_path(self.specs, scene)
        normalize_scale = self.specs.get("normalize_scale")
        pcd1_complete = geometry_utils.read_point_cloud(self.geometries_path.get("pcd1_complete"))
        pcd2_complete = geometry_utils.read_point_cloud(self.geometries_path.get("pcd2_complete"))
        pcd1_partial = geometry_utils.read_point_cloud(self.geometries_path.get("pcd1_partial"))
        pcd2_partial = geometry_utils.read_point_cloud(self.geometries_path.get("pcd2_partial"))
        IBS = geometry_utils.read_point_cloud(self.geometries_path.get("IBS"))
        sphere_center, sphere_radius, direction1, direction2 = geometry_utils.read_medial_axis_sphere_total(self.geometries_path.get("Medial_axis_sphere"))

        translate1, scale1 = geometry_utils.get_pcd_normalize_para(pcd1_complete)
        scale1 /= normalize_scale
        pcd1_complete_normalized = geometry_utils.normalize_geometry(pcd1_complete, translate1, scale1)
        pcd1_partial_normalized = geometry_utils.normalize_geometry(pcd1_partial, translate1, scale1)
        IBS1_normalized = geometry_utils.normalize_geometry(IBS, translate1, scale1)
        sphere_center1, sphere_radius1 = geometry_utils.sphere_transform(sphere_center, sphere_radius, translate1, scale1)
        self.save_pcd(pcd1_complete_normalized, self.get_pcd_complete_save_path(scene, '0'))
        self.save_pcd(pcd1_partial_normalized, self.get_pcd_partial_save_path(scene, '0'))
        self.save_pcd(IBS1_normalized, self.get_IBS_save_path(scene, '0'))
        self.save_medial_axis_sphere(sphere_center1, sphere_radius1, direction1, self.get_medial_axis_sphere__save_path(scene, '0'))
        self.save_normalize_para(translate1, scale1, self.get_normalize_para_save_path(scene, '0'))

        translate2, scale2 = geometry_utils.get_pcd_normalize_para(pcd2_complete)
        scale2 /= normalize_scale
        pcd2_complete_normalized = geometry_utils.normalize_geometry(pcd2_complete, translate2, scale2)
        pcd2_partial_normalized = geometry_utils.normalize_geometry(pcd2_partial, translate2, scale2)
        IBS2_normalized = geometry_utils.normalize_geometry(IBS, translate2, scale2)
        sphere_center2, sphere_radius2 = geometry_utils.sphere_transform(sphere_center, sphere_radius, translate2, scale2)
        self.save_pcd(pcd2_complete_normalized, self.get_pcd_complete_save_path(scene, '1'))
        self.save_pcd(pcd2_partial_normalized, self.get_pcd_partial_save_path(scene, '1'))
        self.save_pcd(IBS2_normalized, self.get_IBS_save_path(scene, '1'))
        self.save_medial_axis_sphere(sphere_center2, sphere_radius2, direction2, self.get_medial_axis_sphere__save_path(scene, '1'))
        self.save_normalize_para(translate2, scale2, self.get_normalize_para_save_path(scene, '1'))


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
    config_filepath = 'configs/get_normalize_data.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("pcd_partial_dir"))
    path_utils.generate_path(specs.get("path_options").get("pcd_complete_save_dir"))
    path_utils.generate_path(specs.get("path_options").get("pcd_partial_save_dir"))
    path_utils.generate_path(specs.get("path_options").get("IBS_save_dir"))
    path_utils.generate_path(specs.get("path_options").get("Medial_axis_save_dir"))
    path_utils.generate_path(specs.get("path_options").get("normalize_para_save_dir"))

    logger = logging.getLogger("get_IBS")
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

            trainDataGenerator = TrainDataGenerator(specs, _logger)
            trainDataGenerator.handle_scene(filename)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)
