"""
从Mesh上采集点云
"""
import logging
import multiprocessing
import os
import re

import open3d as o3d
from utils import path_utils, log_utils


def save_pcd(specs, scene, pcd:tuple):
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


class TrainDataGenerator:
    def __init__(self, specs):
        self.specs = specs
        self.geometries_path = None

    def handle_scene(self, scene):
        self.geometries_path = path_utils.get_geometries_path(self.specs, scene)

        sample_num = self.specs["sample_num"]
        mesh1 = o3d.io.read_triangle_mesh(self.geometries_path["mesh1"])
        mesh2 = o3d.io.read_triangle_mesh(self.geometries_path["mesh2"])
        pcd1 = mesh1.sample_points_poisson_disk(sample_num)
        pcd2 = mesh2.sample_points_poisson_disk(sample_num)

        save_pcd(self.specs, scene, (pcd1, pcd2))


def my_process(scene, specs):
    _logger = logging.getLogger()
    _logger.setLevel("INFO")
    file_handler = logging.FileHandler("logs/get_pcd_from_mesh/{}.log".format(scene), mode="w")
    file_handler.setLevel(level=logging.INFO)
    _logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    _logger.addHandler(stream_handler)

    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")

    # 其他任务操作
    trainDataGenerator = TrainDataGenerator(specs)

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

    pool = multiprocessing.Pool(processes=10)
    file_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                file_list.append(filename)

    trainDataGenerator = TrainDataGenerator(specs)
    for file in file_list:
        trainDataGenerator.handle_scene(file)
        pool.apply_async(my_process, (file, specs,))

    pool.close()
    pool.join()
