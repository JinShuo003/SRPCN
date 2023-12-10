"""
从成对的Mesh采集具有遮挡关系的残缺点云
"""

import json
import logging
import math
import multiprocessing
import os
import re

import numpy as np
import open3d as o3d

from utils import path_utils, geometry_utils, random_utils, log_utils

logger = None


class SampleMethodException(Exception):
    def __init__(self, message="Illegal sample method, surface or IOU are supported"):
        self.message = message
        super.__init__(message)


class BorderNotSetException(Exception):
    def __init__(self):
        self.msg = "Border haven't been set"


class DirectionNotSetException(Exception):
    def __init__(self):
        self.msg = "Direction is None, you should set border first, the direction will be computed automatically"


class DataTypeInvalidException(Exception):
    def __init__(self, type: str):
        self.msg = "The type of the point should be {}".format(type)


class DataDemensionInvalidException(Exception):
    def __init__(self, dimension):
        self.msg = "The demension of data should be {}".format(dimension)


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def getGeometriesPath(specs, scene):
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    category = re.match(category_re, scene).group()
    scene = re.match(scene_re, scene).group()

    geometries_path = dict()

    mesh_dir = specs["mesh_dir"]

    mesh1_filename = '{}_{}.obj'.format(scene, 0)
    mesh2_filename = '{}_{}.obj'.format(scene, 1)

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)

    return geometries_path


def save_pcd(specs, pcd1_list, pcd2_list, view_index_list, scene):
    pcd_dir = specs['pcd_partial_save_dir']
    category = re.match(specs['category_re'], scene).group()

    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(pcd_dir, category)):
        os.makedirs(os.path.join(pcd_dir, category))

    for i in range(len(pcd1_list)):
        # 获取点云名
        pcd1_filename = '{}_view{}_0.ply'.format(scene, view_index_list[i])
        pcd2_filename = '{}_view{}_1.ply'.format(scene, view_index_list[i])

        # 保存点云
        pcd1_path = os.path.join(pcd_dir, category, pcd1_filename)
        if os.path.isfile(pcd1_path):
            os.remove(pcd1_path)
        pcd2_path = os.path.join(pcd_dir, category, pcd2_filename)
        if os.path.isfile(pcd2_path):
            os.remove(pcd2_path)
        o3d.io.write_point_cloud(pcd1_path, pcd1_list[i])
        o3d.io.write_point_cloud(pcd2_path, pcd2_list[i])


class Plane:
    def __init__(self):
        # 平面的四个角点，左上、左下、右上、右下
        self.border: tuple = None
        # 平面的四个方向向量，左、右、上、下
        self.direction: tuple = None

    def set_border(self, border: tuple):
        required_border_type = tuple
        required_border_size = 4
        required_point_type = np.ndarray
        required_point_shape = 3,
        if not isinstance(border, required_border_type):
            raise DataTypeInvalidException(required_border_type)
        if not border.__len__() == required_border_size:
            raise DataDemensionInvalidException(required_border_size)
        for point in border:
            if not isinstance(point, required_point_type):
                raise DataTypeInvalidException()
            if not point.shape == required_point_shape:
                raise DataDemensionInvalidException(required_point_shape)
        self.border = border
        self._compute_direction()

    def get_border(self):
        if self.border is None:
            raise BorderNotSetException()
        return self.border

    def get_left_up(self):
        if self.border is None:
            raise BorderNotSetException()
        return self.border[0]

    def get_left_down(self):
        if self.border is None:
            raise BorderNotSetException()
        return self.border[1]

    def get_right_up(self):
        if self.border is None:
            raise BorderNotSetException()
        return self.border[2]

    def get_right_down(self):
        if self.border is None:
            raise BorderNotSetException()
        return self.border[3]

    def _compute_direction(self):
        if self.border is None:
            raise BorderNotSetException()
        left_up = self.get_left_up()
        left_down = self.get_left_down()
        right_up = self.get_right_up()
        right_down = self.get_right_down()
        dir_left = (left_up - right_up) / np.linalg.norm(left_up - right_up)
        dir_right = (right_up - left_up) / np.linalg.norm(right_up - left_up)
        dir_up = (left_up - left_down) / np.linalg.norm(left_up - left_down)
        dir_down = (left_down - left_up) / np.linalg.norm(left_down - left_up)
        self.direction = (dir_left, dir_right, dir_up, dir_down)

    def get_dir_left(self):
        if self.direction is None:
            raise DirectionNotSetException()
        return self.direction[0]

    def get_dir_right(self):
        if self.direction is None:
            raise DirectionNotSetException()
        return self.direction[1]

    def get_dir_up(self):
        if self.direction is None:
            raise DirectionNotSetException()
        return self.direction[2]

    def get_dir_down(self):
        if self.direction is None:
            raise DirectionNotSetException()
        return self.direction[3]


class ScanPcdGenerator:
    def __init__(self, specs, visualize, mesh1, mesh2):
        self.specs = specs
        self.visualize = visualize
        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.mesh1_triangels = np.asarray(mesh1.triangles)
        self.mesh2_triangels = np.asarray(mesh2.triangles)
        self.mesh1_vertices = np.asarray(mesh1.vertices)
        self.mesh2_vertices = np.asarray(mesh2.vertices)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.get_ray_casting_scene()
        self.scan_plane = Plane()
        self.resolution_width = self.specs["scan_options"]["resolution_width"]
        self.resolution_height = self.specs["scan_options"]["resolution_height"]
        self.pixel_width = 0
        self.pixel_height = 0

    def get_ray_casting_scene(self):
        """初始化光线追踪场景"""
        mesh1_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh1)
        mesh2_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh2)
        self.scene.add_triangles(mesh1_t)
        self.scene.add_triangles(mesh2_t)

    def get_pixel_size(self, plane: Plane, widthResolution: int, heightResolution: int):
        """
        计算出当前平面尺寸和宽高分辨率下，相邻射线投影点在宽、高方向上的距离
        plane: 射线投影平面
        widthResolution: 宽方向上的射线总个数
        heightResolution: 高方向上的射线总个数
        射线总数为n，需要将length分为n-1份
        """
        pixelWidth = np.linalg.norm(plane.get_left_up() - plane.get_right_up()) / (widthResolution - 1)
        pixelHeight = np.linalg.norm(plane.get_left_up() - plane.get_left_down()) / (heightResolution - 1)
        return pixelWidth, pixelHeight

    def build_plane(self, rays):
        # 根据光线构造虚平面
        plane = Plane()
        plane.set_border(self.get_border_points(rays))
        return plane

    def get_projection_points(self, theta, phi, r, fov_deg):
        """
        Args:
            theta: 球坐标天顶角
            phi: 球坐标方位角
            r: 相机所在球的半径
            fov_deg: 视场角
        Returns:
            光线在投影平面上的投影点
            光线，open3d.Tensor
        """
        # 视点
        theta_radian = math.radians(theta)
        phi_radian = math.radians(phi)

        eye = [r * math.sin(theta_radian) * math.cos(phi_radian),
               r * math.cos(theta_radian),
               r * math.sin(theta_radian) * math.sin(phi_radian)]
        # 将视点朝向(0, 0, 0)，发射8*8条光线
        rays = self.scene.create_rays_pinhole(fov_deg=fov_deg,
                                              center=[0, 0, 0],
                                              eye=eye,
                                              up=[0, 1, 0],
                                              width_px=8,
                                              height_px=8)
        # 根据光线构造出一个虚拟平面，用于后续计算射线
        self.scan_plane = self.build_plane(rays)
        # 根据虚拟平面的尺寸和设置的分辨率，计算像素尺寸
        self.pixel_width, self.pixel_height = self.get_pixel_size(self.scan_plane, self.resolution_width,
                                                                  self.resolution_height)

        # 求出虚平面上所有光线的投影点
        points = []
        # 第一行
        row0_points = []
        for i in range(self.resolution_width):
            point = self.scan_plane.get_left_up() + i * self.pixel_width * self.scan_plane.get_dir_right()
            row0_points.append(point)
        # 将第一行向下扩展
        for point in row0_points:
            for i in range(self.resolution_height):
                point_ = point + i * self.pixel_height * self.scan_plane.get_dir_down()
                points.append(point_)

        eye = np.array(eye).reshape(1, 3)
        points = np.array(points).reshape(-1, 3)
        return eye, points

    def get_rays_from_projection_points(self, eye, points):
        # 构造open3d ray (eye, direction): Tensor(n)
        rays = []
        for i in range(points.shape[0]):
            direction = (points[i] - eye) / np.linalg.norm((points[i] - eye))
            rays.append(np.concatenate((eye, direction), axis=1).reshape(6))
        return o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)

    def get_border_points(self, rays):
        rays_ = rays.numpy()[:, :, 3:6]
        row, col, _ = rays_.shape
        left_up = rays_[row - 1][col - 1]
        left_down = rays_[0][col - 1]
        right_up = rays_[row - 1][0]
        right_down = rays_[0][0]
        return left_up, left_down, right_up, right_down

    def visualize_rays_from_projection_points(self, points: np.ndarray, eye: np.ndarray):
        required_data_type = np.ndarray
        required_data_dimension = (1, 3)
        if not isinstance(eye, required_data_type):
            raise DataTypeInvalidException(required_data_type)
        if not isinstance(points, required_data_type):
            raise DataTypeInvalidException(required_data_type)
        if eye.shape != required_data_dimension:
            raise DataDemensionInvalidException(required_data_dimension)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise DataDemensionInvalidException("n*3")

        points = np.concatenate((eye, points), axis=0)
        lines = [[0, i] for i in range(1, points.shape[0] - 1)]
        colors = [[1, 0, 0] for i in range(lines.__len__())]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(points)
        mesh_sphere = geometry_utils.get_sphere_pcd()
        o3d.visualization.draw_geometries([lines_pcd, mesh_sphere, self.mesh1, self.mesh2])

    def visualize_rays(self, eye, rays):
        # 可视化光线，从eye为起点，沿着射线方向移动d，得到射线上另一点
        points = [eye]
        rays_np = rays.numpy()
        for i in range(rays_np.shape[0]):
            points.append(eye + 2 * rays_np[i][3:6])
        points = np.array(points).reshape(-1, 3)
        lines = [[0, i] for i in range(1, points.shape[0])]
        colors = [[1, 0, 0] for i in range(lines.__len__())]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(points)
        mesh_sphere = geometry_utils.get_sphere_pcd()
        o3d.visualization.draw_geometries([lines_pcd, mesh_sphere, self.mesh1, self.mesh2])

    def get_ray_cast_result(self, rays):
        return self.scene.cast_rays(rays)

    def expand_points_in_rectangle(self, points):
        expand_points_num = self.specs["scan_options"]["expand_points_num"]
        # 根据当前的像素大小扩展投影点
        expanded_points = []
        # 对于每一个点：
        # 以该点为中心，以当前像素尺寸得到一个像素，在像素内随机扩充
        for i in range(points.shape[0]):
            x_list = random_utils.randNormalFloat(-self.pixel_width / 2, self.pixel_width / 2, expand_points_num)
            y_list = random_utils.randNormalFloat(-self.pixel_height / 2, self.pixel_height / 2, expand_points_num)
            expanded_points.append(points[i])
            for j in range(expand_points_num):
                expanded_points.append(points[i]
                                       + self.scan_plane.get_dir_right() * x_list[j]
                                       + self.scan_plane.get_dir_up() * y_list[j])
        return np.array(expanded_points).reshape(-1, 3)

    def expand_points_on_border_of_pixel(self, points):
        expand_points_num = self.specs["scan_options"]["expand_points_num"]
        # 根据当前的像素大小扩展投影点
        expanded_points = []
        # 对于每一个点：
        # 以该点为中心，以当前像素尺寸得到一个像素，在像素的四条边上随机选取一个点
        for i in range(points.shape[0]):
            x_list = random_utils.randNormalFloat(-self.pixel_width / 2, self.pixel_width / 2, 2)
            y_list = random_utils.randNormalFloat(-self.pixel_height / 2, self.pixel_height / 2, 2)
            expanded_points.append(points[i])
            expanded_points.append(
                points[i] + self.scan_plane.get_dir_left() * self.pixel_width / 2 + self.scan_plane.get_dir_up() *
                y_list[0])
            expanded_points.append(
                points[i] + self.scan_plane.get_dir_right() * self.pixel_width / 2 + self.scan_plane.get_dir_up() *
                y_list[1])
            expanded_points.append(
                points[i] + self.scan_plane.get_dir_up() * self.pixel_height / 2 + self.scan_plane.get_dir_right() *
                x_list[0])
            expanded_points.append(
                points[i] + self.scan_plane.get_dir_down() * self.pixel_height / 2 + self.scan_plane.get_dir_right() *
                x_list[1])
        return np.array(expanded_points).reshape(-1, 3)

    def get_points_intersect(self, projection_points: np.ndarray, cast_result):
        geometry_ids = cast_result["geometry_ids"].numpy()
        points_intersect_with_obj1 = []
        points_intersect_with_obj2 = []
        for ray_id, intersect_id in enumerate(geometry_ids):
            if intersect_id == 0:
                points_intersect_with_obj1.append(projection_points[ray_id])
            if intersect_id == 1:
                points_intersect_with_obj2.append(projection_points[ray_id])
        points_intersect_with_obj1 = np.array(points_intersect_with_obj1).reshape(-1, 3)
        points_intersect_with_obj2 = np.array(points_intersect_with_obj2).reshape(-1, 3)
        return points_intersect_with_obj1, points_intersect_with_obj2

    def is_view_legal(self, points_obj1, points_obj2, cast_result, min_init_point_num, min_init_radius):
        """
        判断当前射线求交结果是否满足要求
        Args:
            points_obj1: 与obj1相交的射线在投影平面上投影点
            points_obj2: 与obj2相交的射线在投影平面上投影点
            cast_result: 射线求交结果
            legal_point_num: 最小能接受的相交点
            legal_radius: 最小能接受的外接球半径
        Returns:
            当前求交结果是否满足要求
        """
        if points_obj1.shape[0] == 0 or points_obj2.shape[0] == 0:
            return False
        pcd1, pcd2 = self.get_cur_view_pcd(cast_result)
        if points_obj1.shape[0] < min_init_point_num:
            logger.debug(f"points_obj1 not enough, points_num: {points_obj1.shape[0]}")
            centroid, diameter = geometry_utils.get_pcd_normalize_para(pcd1)
            if diameter/2 < min_init_radius:
                logger.debug(f"points_obj1 radius too small, radius: {diameter/2}")
                return False
        if points_obj2.shape[0] < min_init_point_num:
            logger.debug(f"points_obj2 not enough, points_num: {points_obj2.shape[0]}")
            centroid, diameter = geometry_utils.get_pcd_normalize_para(pcd2)
            if diameter/2 < min_init_radius:
                logger.debug(f"points_obj2 radius too small, radius: {diameter/2}")
                return False
        return True

    def get_current_view_scan_pcd(self, theta, phi):
        scan_options = self.specs["scan_options"]
        camera_ridius = scan_options["camera_ridius"]
        fov_deg = scan_options["fov_deg"]
        min_init_point_num = scan_options["min_init_point_num"]
        min_init_radius = scan_options["min_init_radius"]
        pcd_point_num = scan_options["points_num"]
        pcd_sample_num = 1.5 * pcd_point_num
        assert pcd_sample_num > pcd_point_num

        # 按照配置分辨率获取初始光线的相关信息
        eye, projection_points = self.get_projection_points(theta=theta,
                                                            phi=phi,
                                                            r=camera_ridius,
                                                            fov_deg=fov_deg)
        rays = self.get_rays_from_projection_points(eye, projection_points)
        cast_result = self.get_ray_cast_result(rays)
        points_obj1, points_obj2 = self.get_points_intersect(projection_points, cast_result)
        logger.info("init rays num: {}, intersect with obj1: {}, intersect with obj2: {}"
                     .format(rays.shape[0], points_obj1.shape[0], points_obj2.shape[0]))
        # 如果初始射线与模型的交点小于阈值，且交点的外接球半径非常小，则说明遮挡过于严重，直接舍弃
        if not self.is_view_legal(points_obj1, points_obj2, cast_result, min_init_point_num, min_init_radius):
            logger.warning("not enough init points, theta: {}, phi: {}".format(theta, phi))
            return None, None, False
        rays_obj1 = self.get_rays_from_projection_points(eye, points_obj1)
        rays_obj2 = self.get_rays_from_projection_points(eye, points_obj2)

        # 细分射线，直到足够的射线与两模型相交
        while points_obj1.shape[0] < pcd_sample_num or points_obj2.shape[0] < pcd_sample_num:
            # 与某个物体相交的光线数量不够，则将原有光线在投影平面上进行扩充
            if points_obj1.shape[0] < pcd_sample_num:
                logger.info("intersect points with obj1 not enough, cur: {}, target: {}"
                            .format(points_obj1.shape[0], pcd_sample_num))
                points_obj1 = self.expand_points_in_rectangle(points_obj1)
                rays_obj1 = self.get_rays_from_projection_points(eye, points_obj1)
            if points_obj2.shape[0] < pcd_sample_num:
                logger.info("intersect points with obj2 not enough, cur: {}, target: {}"
                            .format(points_obj2.shape[0], pcd_sample_num))
                points_obj2 = self.expand_points_in_rectangle(points_obj2)
                rays_obj2 = self.get_rays_from_projection_points(eye, points_obj2)
            # 每次都获取新的rays，重新进行光线求交
            rays = np.concatenate((rays_obj1.numpy(), rays_obj2.numpy()), axis=0)
            rays = o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)
            cast_result = self.get_ray_cast_result(rays)
            points_obj1, points_obj2 = self.get_points_intersect(np.concatenate((points_obj1, points_obj2), axis=0),
                                                                 cast_result)
            rays_obj1 = self.get_rays_from_projection_points(eye, points_obj1)
            rays_obj2 = self.get_rays_from_projection_points(eye, points_obj2)

        # 多采集一些点，然后用fps保证均匀性
        pcd1, pcd2 = self.get_cur_view_pcd(cast_result)
        pcd1 = pcd1.farthest_point_down_sample(pcd_point_num)
        pcd2 = pcd2.farthest_point_down_sample(pcd_point_num)
        pcd1.paint_uniform_color((0, 0, 1))
        pcd2.paint_uniform_color((0, 1, 0))

        sphere = geometry_utils.get_sphere_pcd()
        coor = geometry_utils.get_unit_coordinate()
        view_direction = self.get_rays_visualization_single_view(rays)
        o3d.visualization.draw_geometries([pcd1, pcd2, sphere, view_direction, self.mesh1, self.mesh2]
                                          , mesh_show_wireframe=True)

        return pcd1, pcd2, True

    def generate_scan_pcd(self):
        pcd1_partial_list = []
        pcd2_partial_list = []
        scan_view_list = []
        # 球坐标，theta为天顶角，phi为方位角
        index = 0
        for theta in [45, 90, 135]:
            for phi in range(0, 360, 45):
                logger.info("begin generate theta: {}, phi: {}".format(theta, phi))
                pcd1_scan, pcd2_scan, success = self.get_current_view_scan_pcd(theta, phi)
                if not success:
                    continue
                pcd1_partial_list.append(pcd1_scan)
                pcd2_partial_list.append(pcd2_scan)
                scan_view_list.append(index)
                index += 1

        return pcd1_partial_list, pcd2_partial_list, scan_view_list

    def get_cur_view_pcd(self, cast_result):
        hit = cast_result['t_hit'].numpy()
        geometry_ids = cast_result["geometry_ids"].numpy()
        primitive_ids = cast_result["primitive_ids"].numpy()
        primitive_uvs = cast_result["primitive_uvs"].numpy()

        points_pcd1 = []
        points_pcd2 = []

        # 获取光线击中的点
        for i in range(hit.shape[0]):
            if not math.isinf(hit[i]):
                if geometry_ids[i] == 0:
                    points_pcd1.append(
                        self.get_real_coordinate(self.mesh1_vertices, self.mesh1_triangels[primitive_ids[i]],
                                                 primitive_uvs[i]))
                if geometry_ids[i] == 1:
                    points_pcd2.append(
                        self.get_real_coordinate(self.mesh2_vertices, self.mesh2_triangels[primitive_ids[i]],
                                                 primitive_uvs[i]))

        pcd_sample_options = self.specs["scan_options"]
        pcd1_scan = o3d.geometry.PointCloud()
        pcd2_scan = o3d.geometry.PointCloud()
        pcd1_scan.points = o3d.utility.Vector3dVector(points_pcd1)
        pcd2_scan.points = o3d.utility.Vector3dVector(points_pcd2)

        return pcd1_scan, pcd2_scan

    def get_real_coordinate(self, vertices, triangles, uv_coordinate):
        # 将三角形的重心坐标变换为真实坐标
        point1 = vertices[triangles[0]]
        point2 = vertices[triangles[1]]
        point3 = vertices[triangles[2]]
        return uv_coordinate[0] * point1 + uv_coordinate[1] * point2 + (
                1 - uv_coordinate[0] - uv_coordinate[1]) * point3

    def get_rays_visualization(self, rays):
        """获取所有的光线"""
        rays_ = rays.numpy()
        eye = rays_[0][0:3]
        rays_ = rays_[:, 3:6]
        points = []
        points.append(eye)
        for i in range(rays_.shape[0]):
            points.append(rays_[i] + eye)
        points = np.array(points)
        lines = [[0, i] for i in range(1, points.shape[0])]
        colors = [[1, 0, 0] for i in range(lines.__len__())]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(points)

        return lines_pcd

    def get_rays_visualization_single_view(self, rays):
        """获取eye到坐标原点的连线，表示当前视角的方向向量"""
        eye = rays.numpy()[0][0:3].reshape(1, 3)
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector([[0, 1]])
        lines_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.array([[0., 0., 0.]]), eye)))
        return lines_pcd


class TrainDataGenerator:
    def __init__(self, specs):
        self.specs = specs
        self.geometries_path = None
        # input mesh
        self.mesh1 = None
        self.mesh2 = None
        # single view scan point cloud
        self.pcd1_partial_list = []
        self.pcd2_partial_list = []
        # the view index of scan point cloud
        self.scan_view_list = []

    def get_mesh(self):
        self.mesh1 = o3d.io.read_triangle_mesh(self.geometries_path["mesh1"])
        self.mesh2 = o3d.io.read_triangle_mesh(self.geometries_path["mesh2"])

    def get_init_geometries(self):
        self.get_mesh()

    def get_scan_pcd(self):
        scanPcdGenerator = ScanPcdGenerator(self.specs, self.specs["visualize"], self.mesh1, self.mesh2)
        return scanPcdGenerator.generate_scan_pcd()

    def visualize_result(self, pcd1_partial_list, pcd2_partial_list, scan_view_list):
        coor = geometry_utils.get_unit_coordinate()
        sphere = geometry_utils.get_sphere_pcd()

        for i in range(len(pcd1_partial_list)):
            o3d.visualization.draw_geometries(
                [pcd1_partial_list[i], pcd2_partial_list[i], coor, sphere],
                window_name="{}".format(scan_view_list[i]))

    def handle_scene(self, scene):
        self.geometries_path = getGeometriesPath(self.specs, scene)
        self.get_init_geometries()

        pcd1_partial_list, pcd2_partial_list, scan_view_list = self.get_scan_pcd()

        if self.specs["visualize"]:
            self.visualize_result(pcd1_partial_list, pcd2_partial_list,
                                  scan_view_list)
        # save the result
        save_pcd(self.specs, pcd1_partial_list, pcd2_partial_list, scan_view_list, scene)


def my_process(scene, specs):
    process_name = multiprocessing.current_process().name
    logger.info(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs)

    try:
        trainDataGenerator.handle_scene(scene)
        logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        logger.error("scene: {} failed, exception message: {}".format(scene, e.message))


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/get_scan_pcd.json'
    specs = parseConfig(config_filepath)

    # 日志模块
    logger = log_utils.get_logger(specs.get("log_options"))

    processNum = specs["process_num"]
    # 构建文件树
    filename_tree = path_utils.getFilenameTree(specs, specs["mesh_dir"])
    # 处理文件夹，不存在则创建
    path_utils.generatePath(specs["pcd_partial_save_dir"])

    # 创建进程池，指定进程数量
    pool = multiprocessing.Pool(processes=processNum)

    scene_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            scene_list.append(scene)
    trainDataGenerator = TrainDataGenerator(specs)
    # 使用进程池执行任务，返回结果列表
    for scene in scene_list:
        logger.info("current scene: {}".format(scene))
        # pool.apply_async(my_process, (scene, specs,))
        trainDataGenerator.handle_scene(scene)

    # 关闭进程池
    pool.close()
    pool.join()
