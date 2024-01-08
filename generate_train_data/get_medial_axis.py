"""
1. 将IBS_gt作为中轴，计算IBS_gt点云每个点到两侧点云的距离，并取最小值
2. 以该点为中心，该距离最小值做半径作球
3. 遍历IBS_gt点云的每个点，求出所有球
4. 所有球的集合可以看做填充了两物体之间的空间，若补全结果落在球的范围内，则必然需要惩罚（因为球的获取采取了保守的策略）
缺陷：求出的球集合无法填充整个空间，需要提高密度
思路：已知Medial Axis，如何求出所有填充球？
1. 输入Mesh形式的IBS，然后随机采一批点，计算他们的Occupied Sphere
2. 遍历每一对计算结果，如果两个球相离，则求出球心连线中点，找到该点到Mesh表面的最近点，再次进行计算，将计算结果合并
3. 直到每一对球都相交为止
"""

import os
import re
import open3d as o3d
import numpy as np
import multiprocessing
import logging
import trimesh

from utils import geometry_utils, path_utils, log_utils


class MedialAxisaGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger

    def get_nearest_points(self, query: np.ndarray, mesh: trimesh.Trimesh):
        return abs(trimesh.proximity.signed_distance(mesh, query))

    def get_sphere_list(self, IBS: np.ndarray, distance1, distance2):
        assert IBS.shape[0] == len(distance1)

        sphere_list = list()
        for i in range(IBS.shape[0]):
            radius = min(distance1[i], distance2[i])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere = sphere.sample_points_uniformly(64)
            sphere.paint_uniform_color((1, 0, 0))
            sphere.translate(IBS[i])
            sphere_list.append(sphere)
        return sphere_list

    def visualize_medial_axis(self, IBS, distance1, distance2, mesh1, mesh2, IBS_o3d):
        for i in range(IBS.shape[0]):
            radius = min(distance1[i], distance2[i])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.paint_uniform_color((1, 0, 0))
            sphere.translate(IBS[i])
            o3d.visualization.draw_geometries([mesh1, mesh2, sphere, IBS_o3d], mesh_show_wireframe=True,
                                              mesh_show_back_face=True)

    def save_medial_axis(self, scene: str, center: np.ndarray, radius: np.ndarray):
        assert center.shape[0] == radius.shape[0]

        medial_axis_save_dir = self.specs.get("path_options").get("MedialAxis_save_dir")
        category = re.match(self.specs.get("path_options").get("format_info").get("category_re"), scene).group()
        if not os.path.isdir(os.path.join(medial_axis_save_dir, category)):
            os.makedirs(os.path.join(medial_axis_save_dir, category))

        medial_axis_filename = "{}.npz".format(scene)
        medial_axis_path = os.path.join(medial_axis_save_dir, category, medial_axis_filename)
        np.savez(medial_axis_path, center=center, radius=radius)

    def handle_scene(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])
        IBS = geometry_utils.read_point_cloud(geometries_path["ibs_pcd"])
        IBS_np = np.asarray(IBS.points)
        distance1 = self.get_nearest_points(IBS_np, geometry_utils.o3d2trimesh(mesh1))
        distance2 = self.get_nearest_points(IBS_np, geometry_utils.o3d2trimesh(mesh2))

        center = IBS_np
        radius = np.min((distance1, distance2), axis=0)
        self.save_medial_axis(scene, center, radius)

        # IBS.paint_uniform_color((0, 0, 0))
        # mesh1.paint_uniform_color((0, 1, 0))
        # mesh2.paint_uniform_color((0, 0, 1))
        # self.visualize_medial_axis(IBS_np, distance1, distance2, mesh1, mesh2, IBS)

        # sphere_list = self.get_sphere_list(IBS_np, distance1, distance2)
        # sphere_list.append(mesh1)
        # sphere_list.append(mesh2)
        # sphere_list.append(IBS)
        #
        # o3d.visualization.draw_geometries(sphere_list)


def my_process(scene, specs):
    _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)
    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = MedialAxisaGenerator(specs, _logger)

    try:
        trainDataGenerator.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e))
    finally:
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)


if __name__ == '__main__':
    config_filepath = 'configs/get_medial_axis.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("MedialAxis_save_dir"))

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
                                                                         filename, file_handler_level=logging.DEBUG)

            trainDataGenerator = MedialAxisaGenerator(specs, _logger)
            try:
                trainDataGenerator.handle_scene(filename)
                _logger.info("scene: {} succeed".format(filename))
            except Exception as e:
                _logger.error("scene: {} failed, exception message: {}".format(filename, e))
            finally:
                _logger.removeHandler(file_handler)
                _logger.removeHandler(stream_handler)
