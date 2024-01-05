"""
从IBS上采点
"""
import logging
import multiprocessing
import os
import re
import numpy as np

import open3d as o3d
import trimesh
from utils import path_utils, log_utils, geometry_utils


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

    def subdevide_mesh(self, o3d_obj, max_edge):
        trimesh_obj = geometry_utils.o3d2trimesh(o3d_obj)
        vertices, faces = trimesh.remesh.subdivide_to_size(trimesh_obj.vertices, trimesh_obj.faces, max_edge)
        trimesh_obj = trimesh.Trimesh(vertices, faces, process=True)
        return geometry_utils.trimesh2o3d(trimesh_obj)

    def get_total_bounding_box(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud):
        aabb1 = pcd1.get_axis_aligned_bounding_box()
        aabb2 = pcd2.get_axis_aligned_bounding_box()
        max_border = np.max((aabb1.get_max_bound(), aabb2.get_max_bound()), axis=0)
        min_border = np.min((aabb1.get_min_bound(), aabb2.get_min_bound()), axis=0)
        aabb_total = o3d.geometry.AxisAlignedBoundingBox(min_border, max_border)
        return aabb_total

    def sample_with_weight(self, ibs: o3d.geometry.TriangleMesh, pcd1, pcd2, points_num,
                           weight_distance_exp, weight_angle_threshold):
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        ibs_vertices = np.asarray(ibs.vertices)
        ibs_triangles = np.asarray(ibs.triangles)
        ibs_triangle_num = ibs_triangles.shape[0]
        ibs_triangle_normals = np.asarray(ibs.triangle_normals)
        ibs_triangle_mid_points = np.zeros((ibs_triangle_num, 3))
        ibs_triangle_area = np.zeros(ibs_triangle_num)

        assert ibs_triangle_normals.shape[0] == ibs_triangle_num

        # 计算面片中心点和面片面积
        for i, triangle in enumerate(ibs_triangles):
            p1 = ibs_vertices[triangle[0]]
            p2 = ibs_vertices[triangle[1]]
            p3 = ibs_vertices[triangle[2]]
            e1 = p2 - p1
            e2 = p3 - p1
            ibs_triangle_mid_points[i] = np.mean((p1, p2, p3), axis=0)
            ibs_triangle_area[i] = 0.5 * np.linalg.norm(np.cross(e1, e2))

        weights_area = ibs_triangle_area
        weights_distance = np.zeros(ibs_triangle_num)
        weights_angle = np.zeros(ibs_triangle_num)
        # D: 总aabb框对角线的一半
        aabb_total = self.get_total_bounding_box(pcd1, pcd2)
        D = np.linalg.norm(aabb_total.get_max_bound() - aabb_total.get_min_bound()) / 2
        # 面片中心点到pcd1、pcd2的最小距离
        distance_pcd1 = np.linalg.norm(ibs_triangle_mid_points[:, np.newaxis, :] - points1, axis=2)
        distance_pcd2 = np.linalg.norm(ibs_triangle_mid_points[:, np.newaxis, :] - points2, axis=2)
        min_distance_pcd1 = np.min(distance_pcd1, axis=1)
        min_distance_pcd2 = np.min(distance_pcd2, axis=1)
        min_indices_pcd1 = np.argmin(distance_pcd1, axis=1)
        min_indices_pcd2 = np.argmin(distance_pcd2, axis=1)
        # 计算距离权重和角度权重
        for i in range(ibs_triangle_num):
            # Weight-distance: 面片中心点到点云的最小距离越大，权重越小
            d = min(min_distance_pcd1[i], min_distance_pcd2[i])
            weights_distance[i] = pow((1 - d/D), weight_distance_exp)

            # Weight-angle: 面片中心点与点云最近点连线组成向量与面片法向量的角度越大，权重越小
            closest_point1 = points1[min_indices_pcd1[i]]
            closest_point2 = points2[min_indices_pcd2[i]]
            v1 = closest_point1 - ibs_triangle_mid_points[i]
            v2 = closest_point2 - ibs_triangle_mid_points[i]
            n = ibs_triangle_normals[i]
            cosine_similarity1 = np.dot(v1, n) / (np.linalg.norm(v1) * np.linalg.norm(n))
            cosine_similarity2 = np.dot(v2, n) / (np.linalg.norm(v2) * np.linalg.norm(n))
            theta1 = np.arccos(np.clip(cosine_similarity1, -1.0, 1.0))
            theta2 = np.arccos(np.clip(cosine_similarity2, -1.0, 1.0))
            theta1 = np.degrees(theta1)
            theta2 = np.degrees(theta2)
            theta = min(theta1, theta2)
            theta = theta if 0 < theta < 90 else 180 - theta
            weights_angle[i] = 1 - theta/weight_angle_threshold if theta < weight_angle_threshold else 0

        weights = weights_area * weights_distance * weights_angle
        weights /= sum(weights)
        selected_triangles_idx = np.random.choice(range(ibs_triangle_num), points_num, False, weights)
        sample_points = ibs_triangle_mid_points[selected_triangles_idx]
        ibs_pcd = geometry_utils.get_pcd_from_np(sample_points)

        return ibs_pcd

    def get_ibs_pcd(self, scene):
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        sample_num = self.specs.get("sample_options").get("sample_num")
        subdivide_max_edge = self.specs.get("sample_options").get("subdivide_max_edge")
        weight_distance_exp = self.specs.get("sample_options").get("weight_distance_exp")
        weight_angle_threshold = self.specs.get("sample_options").get("weight_angle_threshold")

        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])
        ibs = geometry_utils.read_mesh(geometries_path["ibs"])
        self.logger.info("ibs with {} vertices, {} triangles".
                         format(np.asarray(ibs.vertices).shape[0], np.asarray(ibs.triangles).shape[0]))
        ibs = self.subdevide_mesh(ibs, subdivide_max_edge)
        self.logger.info("ibs has {} triangles after subdevide".format(np.asarray(ibs.triangles).shape[0]))
        ibs.compute_triangle_normals()

        pcd1 = mesh1.sample_points_poisson_disk(2048)
        pcd2 = mesh2.sample_points_poisson_disk(2048)
        ibs_pcd = self.sample_with_weight(ibs, pcd1, pcd2, sample_num, weight_distance_exp, weight_angle_threshold)
        # pcd1.paint_uniform_color((1, 0, 0))
        # pcd2.paint_uniform_color((0, 1, 0))
        # ibs.paint_uniform_color((0, 0, 1))
        # ibs_pcd.paint_uniform_color((0, 1, 1))
        # o3d.visualization.draw_geometries([pcd1, pcd2, ibs, ibs_pcd], mesh_show_wireframe=True, mesh_show_back_face=True)
        self.logger.info("get {} points from ibs".format(np.asarray(ibs_pcd.points).shape[0]))
        return ibs_pcd

    def handle_scene(self, scene):
        ibs_pcd = self.get_ibs_pcd(scene)
        save_ibs_pcd(self.specs, scene, ibs_pcd)


def my_process(scene, specs):
    _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)
    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs, _logger)

    try:
        trainDataGenerator.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e))
    finally:
        _logger.removeHandler(file_handler)
        _logger.removeHandler(stream_handler)


if __name__ == '__main__':
    config_filepath = 'configs/sample_points_from_mesh.json'
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
            try:
                trainDataGenerator.handle_scene(filename)
                _logger.info("scene: {} succeed".format(filename))
            except Exception as e:
                _logger.error("scene: {} failed, exception message: {}".format(filename, e))
            finally:
                _logger.removeHandler(file_handler)
                _logger.removeHandler(stream_handler)
