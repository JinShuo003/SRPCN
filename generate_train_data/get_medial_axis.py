"""
计算Medial Axis Sphere
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

    def get_closest_points(self, query: np.ndarray, mesh: trimesh.Trimesh):
        cloest1, distance1, triangle_id1 = trimesh.proximity.closest_point(mesh, query)
        return cloest1, abs(distance1), triangle_id1

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

    def get_direction(self, center, cloest1):
        direction = cloest1 - center
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return direction

    def visualize_medial_axis(self, IBS, distance1, distance2, mesh1, mesh2, direction1, direction2, IBS_o3d):
        IBS_np = np.asarray(IBS_o3d.points)
        for i in range(IBS.shape[0]):
            end_point1 = geometry_utils.get_pcd_from_np(direction1[i].reshape(-1, 3))
            end_point2 = geometry_utils.get_pcd_from_np(direction2[i].reshape(-1, 3))
            begin_point = geometry_utils.get_pcd_from_np(IBS_np[i].reshape(-1, 3))
            end_point1.paint_uniform_color((0, 0, 1))
            end_point2.paint_uniform_color((0, 1, 0))
            begin_point.paint_uniform_color((0, 1, 1))

            radius = min(distance1[i], distance2[i])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.paint_uniform_color((1, 0, 0))
            sphere.translate(IBS[i])
            o3d.visualization.draw_geometries([mesh1, mesh2, sphere, IBS_o3d, end_point1, end_point2, begin_point], mesh_show_wireframe=True,
                                              mesh_show_back_face=True)

    def save_medial_axis(self, scene: str, center: np.ndarray, radius: np.ndarray, direction1: np.ndarray, direction2: np.ndarray):
        assert center.shape[0] == radius.shape[0]

        medial_axis_save_dir = self.specs.get("path_options").get("MedialAxis_save_dir")
        category = re.match(self.specs.get("path_options").get("format_info").get("category_re"), scene).group()
        if not os.path.isdir(os.path.join(medial_axis_save_dir, category)):
            os.makedirs(os.path.join(medial_axis_save_dir, category))

        medial_axis_filename = "{}.npz".format(scene)
        medial_axis_path = os.path.join(medial_axis_save_dir, category, medial_axis_filename)
        np.savez(medial_axis_path, center=center, radius=radius, direction1=direction1, direction2=direction2)

    def handle_scene(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])
        IBS = geometry_utils.read_point_cloud(geometries_path["ibs_pcd"])
        IBS_np = np.asarray(IBS.points)
        cloest1, distance1, triangle_id1 = self.get_closest_points(IBS_np, geometry_utils.o3d2trimesh(mesh1))
        cloest2, distance2, triangle_id2 = self.get_closest_points(IBS_np, geometry_utils.o3d2trimesh(mesh2))

        center = IBS_np
        radius = np.min((distance1, distance2), axis=0)
        direction1 = self.get_direction(center, cloest1)
        direction2 = self.get_direction(center, cloest2)
        self.save_medial_axis(scene, center, radius, direction1, direction2)

        # IBS.paint_uniform_color((0, 0, 0))
        # mesh1.paint_uniform_color((0, 1, 0))
        # mesh2.paint_uniform_color((0, 0, 1))
        # self.visualize_medial_axis(IBS_np, distance1, distance2, mesh1, mesh2, cloest1, cloest2, IBS)
        #
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
