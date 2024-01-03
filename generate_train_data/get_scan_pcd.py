"""
从成对的Mesh采集具有遮挡关系的残缺点云
"""
import logging
import math
import multiprocessing
import os
import re

import numpy as np
import open3d as o3d

from utils import path_utils, geometry_utils, random_utils, log_utils, exception_utils

logger = None


class Plane:
    """
    空间平面，用于计算射线
    该平面只有两个旋转自由度，永远不会有滚转角，即矩形的上下边永远平行于x-y平面
    关键参数：矩形的四个边界点、四个方向向量
    """

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
            raise exception_utils.DataTypeInvalidException(required_border_type)
        if not border.__len__() == required_border_size:
            raise exception_utils.DataDemensionInvalidException(required_border_size)
        for point in border:
            if not isinstance(point, required_point_type):
                raise exception_utils.DataTypeInvalidException()
            if not point.shape == required_point_shape:
                raise exception_utils.DataDemensionInvalidException(required_point_shape)
        self.border = border
        self._compute_direction()

    def get_border(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border

    def get_left_up(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[0]

    def get_left_down(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[1]

    def get_right_up(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[2]

    def get_right_down(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[3]

    def _compute_direction(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
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
            raise exception_utils.DirectionNotSetException()
        return self.direction[0]

    def get_dir_right(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[1]

    def get_dir_up(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[2]

    def get_dir_down(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[3]


class Visualizer:
    def __init__(self):
        pass

    def visualize_rays_from_projection_points(self, eye: np.ndarray, points: np.ndarray, geometries):
        """
        可视化光线
        Args:
            eye: 视点
            points: 投影平面上的投影点
            geometries: 其他几何体
        """
        _eye = eye.reshape(1, 3)
        required_data_type = np.ndarray
        required_data_dimension = (1, 3)
        if not isinstance(_eye, required_data_type):
            raise exception_utils.DataTypeInvalidException(required_data_type)
        if not isinstance(points, required_data_type):
            raise exception_utils.DataTypeInvalidException(required_data_type)
        if _eye.shape != required_data_dimension:
            raise exception_utils.DataDemensionInvalidException(required_data_dimension)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise exception_utils.DataDemensionInvalidException("n*3")

        points = np.concatenate((_eye, points), axis=0)
        lines = [[0, i] for i in range(1, points.shape[0] - 1)]
        colors = [[1, 0, 0] for i in range(lines.__len__())]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(points)

        geometries.append(lines_pcd)
        o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True)

    def visualize_rays(self, eye, rays, geometries):
        """
        可视化射线
        Args:
            eye: 视点
            rays: 射线
            geometries: 其他需要显示的几何体
        """
        points = [eye]
        rays_np = rays.numpy()
        for i in range(rays_np.shape[0]):
            points.append(eye + 3 * rays_np[i][3:6])
        points = np.array(points).reshape(-1, 3)
        lines = [[0, i] for i in range(1, points.shape[0])]
        colors = [[1, 0, 0] for i in range(lines.__len__())]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(points)

        geometries.append(lines_pcd)
        o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True)

    def visualize_geometries(self, geometries):
        o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True)

    def get_view_direction(self, rays):
        """获取eye到坐标原点的连线，表示当前视角的方向向量"""
        eye = rays.numpy()[0][0:3].reshape(1, 3)
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector([[0, 1]])
        lines_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.array([[0., 0., 0.]]), eye)))
        return lines_pcd


class ScanPcdGenerator:
    def __init__(self, specs, is_visualize, mesh1, mesh2, logger):
        self.specs = specs
        self.logger = logger
        self.is_visualize = is_visualize
        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.mesh1_triangels = np.asarray(mesh1.triangles)
        self.mesh2_triangels = np.asarray(mesh2.triangles)
        self.mesh1_vertices = np.asarray(mesh1.vertices)
        self.mesh2_vertices = np.asarray(mesh2.vertices)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.get_ray_casting_scene()
        self.scan_plane = None
        self.resolution_width = self.specs["scan_options"]["resolution_width"]
        self.resolution_height = self.specs["scan_options"]["resolution_height"]
        self.pixel_width = 0
        self.pixel_height = 0
        self.visualizer = Visualizer()

    def get_ray_casting_scene(self):
        """初始化光线追踪场景"""
        mesh1_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh1)
        mesh2_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh2)
        self.scene.add_triangles(mesh1_t)
        self.scene.add_triangles(mesh2_t)

    def get_view_point(self, theta, phi, r):
        """
        根据视角信息返回视点坐标
        Args:
            theta: 球坐标天顶角
            phi: 球坐标方位角
            r: 相机所在球的半径
        Returns:
            视点
        """
        theta_radian = math.radians(theta)
        phi_radian = math.radians(phi)

        eye = [r * math.sin(theta_radian) * math.cos(phi_radian),
               r * math.cos(theta_radian),
               r * math.sin(theta_radian) * math.sin(phi_radian)]
        if theta == 0 or theta == 180:
            eye[0] = 1e-8
        return np.array(eye)

    def get_border_points(self, eye, rays):
        """
        获取open3d API产生光线的四个顶点光线，并从视点出发沿着四条射线的方向偏移len=1，得到四个点
        Args:
            eye: 视点
            rays: open3d rays
        Returns:
            tuple
        """
        direction = rays.numpy()[:, :, 3:6]
        row, col, _ = direction.shape
        left_up = eye + 3 * direction[row - 1][col - 1]
        left_down = eye + 3 * direction[0][col - 1]
        right_up = eye + 3 * direction[row - 1][0]
        right_down = eye + 3 * direction[0][0]
        return left_up, left_down, right_up, right_down

    def build_plane(self, eye, rays):
        """根据光线构造虚平面"""
        plane = Plane()
        plane.set_border(self.get_border_points(eye, rays))
        return plane

    def get_projection_plane(self, eye, fov_deg):
        """
        根据视点和视场角获取虚拟投影平面
        Args:
            eye: 视点
            fov_deg: 视场角
        Returns:
            投影平面
        """
        # 视点朝向(0, 0, 0)，发射8*8条光线
        rays = self.scene.create_rays_pinhole(fov_deg=fov_deg,
                                              center=[0, 0, 0],
                                              eye=eye,
                                              up=[0, 1, 0],
                                              width_px=8,
                                              height_px=8)
        return self.build_plane(eye, rays)

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

    def get_projection_points(self, projection_plane: Plane, resolution_width: int, resolution_height: int):
        """
        给定投影平面、宽高方向上的分辨率，计算平面上的所有投影点
        Args:
            projection_plane: 投影平面
            resolution_width: 宽方向上的分辨率，即投影平面的宽方向上期望有多少个点
            resolution_height: 高方向上的分辨率，即投影平面的高方向上期望有多少个点
        Returns:
            投影点，type: np.ndarray，shape: (n, 3)
        """
        pixel_width, pixel_height = self.get_pixel_size(projection_plane, resolution_width, resolution_height)
        projection_points = []
        for i in range(resolution_width):
            for j in range(resolution_height):
                point = projection_plane.get_left_up() + \
                        i * pixel_width * projection_plane.get_dir_right() + \
                        j * pixel_height * projection_plane.get_dir_down()
                projection_points.append(point)

        projection_points = np.array(projection_points).reshape(-1, 3)
        return pixel_width, pixel_height, projection_points

    def get_rays_from_projection_points(self, eye, projection_points):
        """
        通过视点和投影平面上的投影点构造open3d射线
        Args:
            eye: 视点
            projection_points:
        Returns:
            open3d rays, type: open3d.Tensor
        """
        rays = []
        _eye = eye.reshape(1, 3)
        for i in range(projection_points.shape[0]):
            direction = (projection_points[i] - _eye) / np.linalg.norm((projection_points[i] - _eye))
            rays.append(np.concatenate((_eye, direction), axis=1).reshape(6))
        return o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)

    def get_ray_cast_result(self, scene, rays):
        """获取光线投射结果"""
        return scene.cast_rays(rays)

    def get_points_intersect(self, projection_points: np.ndarray, cast_result):
        """
        从光线在投影平面的投影点中筛选出和和obj1、obj2相交的那部分
        Args:
            projection_points: 所有投影点
            cast_result: 射线求交的结果
        Returns:
            与obj1、obj2相交的点
        """
        assert projection_points.shape[0] == cast_result["t_hit"].shape[0]

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

    def is_view_legal(self, points_obj1, points_obj2):
        """
        判定当前射线求交结果是否能够满足采集数据的要求
        Args:
            points_obj1: 与obj1相交的射线投影点
            points_obj2: 与obj2相交的射线投影点
        Returns:
            是否满足采集条件
        """
        if points_obj1.shape[0] == 0 or points_obj2.shape[0] == 0:
            return False
        return True

    def expand_points_in_rectangle(self, expand_points_num, width, height, plane: Plane, points):
        """
            在每个点的某个邻域内随机采集一些点
        Args:
            expand_points_num: 随机点的数量
            width: 矩形区域的宽
            height: 矩形区域的高
            plane: 随机点所处的平面
            points: 原始点
        Returns:
            扩展后的点，type: np.ndarray，shape: (n, 3)
        """
        expanded_points = []
        for i in range(points.shape[0]):
            x_list = random_utils.randNormalFloat(-width, width, expand_points_num)
            y_list = random_utils.randNormalFloat(-height, height, expand_points_num)
            expanded_points.append(points[i])
            for j in range(expand_points_num):
                expanded_points.append(points[i]
                                       + plane.get_dir_right() * x_list[j]
                                       + plane.get_dir_up() * y_list[j])
        return np.array(expanded_points).reshape(-1, 3)

    def get_real_coordinate(self, vertices, triangles, uv_coordinate):
        # 将三角形的重心坐标变换为真实坐标
        point1 = vertices[triangles[0]]
        point2 = vertices[triangles[1]]
        point3 = vertices[triangles[2]]
        return uv_coordinate[0] * point1 + uv_coordinate[1] * point2 + (
                1 - uv_coordinate[0] - uv_coordinate[1]) * point3

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

        pcd1_scan = o3d.geometry.PointCloud()
        pcd2_scan = o3d.geometry.PointCloud()
        pcd1_scan.points = o3d.utility.Vector3dVector(points_pcd1)
        pcd2_scan.points = o3d.utility.Vector3dVector(points_pcd2)

        return pcd1_scan, pcd2_scan

    def get_current_view_scan_pcd(self, theta, phi):
        """
        获取某个角度观察的残缺点云数据
        Args:
            theta: 球坐标天顶角
            phi: 球坐标方位角
        Returns:
            残缺点云1，残缺点云2，是否采集成功
        """
        scan_options = self.specs["scan_options"]
        camera_ridius = scan_options["camera_ridius"]
        fov_deg = scan_options["fov_deg"]
        pcd_point_num = scan_options["points_num"]
        expand_points_num = scan_options["expand_points_num"]
        pcd_sample_num = 1.5 * pcd_point_num
        assert pcd_sample_num > pcd_point_num

        # 按照配置分辨率获取初始光线的相关信息
        eye = self.get_view_point(theta, phi, camera_ridius)  # 视点，(3)
        self.scan_plane = self.get_projection_plane(eye=eye, fov_deg=fov_deg)  # 投影平面
        self.pixel_width, self.pixel_height, projection_points = \
            self.get_projection_points(self.scan_plane,
                                       self.resolution_width,
                                       self.resolution_height)  # 投影点
        projection_points = self.expand_points_in_rectangle(expand_points_num,
                                                            self.pixel_width,
                                                            self.pixel_height,
                                                            self.scan_plane,
                                                            projection_points)  # 扩充投影点，保证随机性
        rays = self.get_rays_from_projection_points(eye, projection_points)  # 射线
        # self.visualizer.visualize_rays(eye, rays, [self.mesh1, self.mesh2, geometry_utils.get_sphere_pcd(radius=0.5)])
        cast_result = self.get_ray_cast_result(self.scene, rays)  # 射线求交结果
        points_obj1, points_obj2 = self.get_points_intersect(projection_points, cast_result)  # 与obj1、obj2相交的射线投影点
        self.logger.info("init rays num: {}, intersect with obj1: {}, intersect with obj2: {}"
                         .format(rays.shape[0], points_obj1.shape[0], points_obj2.shape[0]))
        # 判断初始结果是否满足采集条件
        if not self.is_view_legal(points_obj1, points_obj2):
            self.logger.warning("not enough init points, theta: {}, phi: {}".format(theta, phi))
            return None, None, False
        rays_obj1 = self.get_rays_from_projection_points(eye, points_obj1)
        rays_obj2 = self.get_rays_from_projection_points(eye, points_obj2)

        # 迭代地在相交的局部增大分辨率，直到足够的射线与两模型相交
        while points_obj1.shape[0] < pcd_sample_num or points_obj2.shape[0] < pcd_sample_num:
            # 与某个物体相交的光线数量不够，则将原有光线在投影平面上进行扩充
            if points_obj1.shape[0] < pcd_sample_num:
                self.logger.info("intersect points with obj1 not enough, cur: {}, target: {}"
                                 .format(points_obj1.shape[0], pcd_sample_num))
                points_obj1 = self.expand_points_in_rectangle(expand_points_num,
                                                              self.pixel_width,
                                                              self.pixel_height,
                                                              self.scan_plane,
                                                              points_obj1)
                rays_obj1 = self.get_rays_from_projection_points(eye, points_obj1)
            if points_obj2.shape[0] < pcd_sample_num:
                self.logger.info("intersect points with obj2 not enough, cur: {}, target: {}"
                                 .format(points_obj2.shape[0], pcd_sample_num))
                points_obj2 = self.expand_points_in_rectangle(expand_points_num,
                                                              self.pixel_width,
                                                              self.pixel_height,
                                                              self.scan_plane,
                                                              points_obj2)
                rays_obj2 = self.get_rays_from_projection_points(eye, points_obj2)
                # self.visualize_rays(eye, rays_obj2)
            # 每轮都将来自obj1和来自obj2的射线拼接在一起，重新进行射线求交
            rays = np.concatenate((rays_obj1.numpy(), rays_obj2.numpy()), axis=0)
            rays = o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)
            cast_result = self.get_ray_cast_result(self.scene, rays)
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

        return pcd1, pcd2, True

    def generate_scan_pcd(self):
        pcd1_partial_list = []
        pcd2_partial_list = []
        scan_view_list = []
        # 球坐标，theta为天顶角，phi为方位角
        index = 0
        # 俯视
        self.logger.info("\n")
        self.logger.info("begin generate theta: {}, phi: {}".format(0, 0))
        pcd1_scan, pcd2_scan, success = self.get_current_view_scan_pcd(0, 0)
        if success:
            pcd1_partial_list.append(pcd1_scan)
            pcd2_partial_list.append(pcd2_scan)
            scan_view_list.append(index)
            index += 1
        # 仰视
        self.logger.info("\n")
        self.logger.info("begin generate theta: {}, phi: {}".format(180, 0))
        pcd1_scan, pcd2_scan, success = self.get_current_view_scan_pcd(180, 0)
        if success:
            pcd1_partial_list.append(pcd1_scan)
            pcd2_partial_list.append(pcd2_scan)
            scan_view_list.append(index)
            index += 1
        for theta in [45, 90, 135]:
            for phi in range(0, 360, 45):
                self.logger.info("\n")
                self.logger.info("begin generate theta: {}, phi: {}".format(theta, phi))
                pcd1_scan, pcd2_scan, success = self.get_current_view_scan_pcd(theta, phi)
                if success:
                    pcd1_partial_list.append(pcd1_scan)
                    pcd2_partial_list.append(pcd2_scan)
                    scan_view_list.append(index)
                    index += 1

        return pcd1_partial_list, pcd2_partial_list, scan_view_list


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger
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
        scanPcdGenerator = ScanPcdGenerator(self.specs, self.specs["visualize"], self.mesh1, self.mesh2, self.logger)
        return scanPcdGenerator.generate_scan_pcd()

    def visualize_result(self, pcd1_partial_list, pcd2_partial_list, scan_view_list):
        coor = geometry_utils.get_coordinate()
        sphere = geometry_utils.get_sphere_pcd()

        for i in range(len(pcd1_partial_list)):
            o3d.visualization.draw_geometries(
                [pcd1_partial_list[i], pcd2_partial_list[i], coor, sphere],
                window_name="{}".format(scan_view_list[i]))

    def save_pcd(self, specs, pcd1_list, pcd2_list, view_index_list, scene):
        """
        保存点云数据
        Args:
            specs: 配置信息
            pcd1_list: obj1各个角度的partial point cloud
            pcd2_list: obj2各个角度的partial point cloud
            view_index_list: 每个视角的index
            scene: 当前场景名
        """
        assert len(pcd1_list) == len(pcd2_list) and len(pcd1_list) == len(view_index_list)

        pcd_dir = specs.get("path_options").get("pcd_partial_save_dir")
        category = re.match(specs.get("path_options").get("format_info").get("category_re"), scene).group()

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

    def handle_scene(self, scene):
        self.geometries_path = path_utils.get_geometries_path(self.specs, scene)
        self.get_init_geometries()

        pcd1_partial_list, pcd2_partial_list, scan_view_list = self.get_scan_pcd()

        if self.specs["visualize"]:
            self.visualize_result(pcd1_partial_list, pcd2_partial_list, scan_view_list)

        self.save_pcd(self.specs, pcd1_partial_list, pcd2_partial_list, scan_view_list, scene)
        self.logger.info("current scene saved successfully")


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
    config_filepath = 'configs/get_scan_pcd.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("pcd_partial_save_dir"))

    logger = logging.getLogger("get_IBS")
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    scene_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            scene_list.append(scene)

    if specs.get("use_process_poll"):
        pool = multiprocessing.Pool(processes=specs.get("process_num"))

        for scene in scene_list:
            logger.info("current scene: {}".format(scene))
            pool.apply_async(my_process, (scene, specs,))

        pool.close()
        pool.join()
    else:
        for scene in scene_list:
            logger.info("current scene: {}".format(scene))
            _logger, file_handler, stream_handler = log_utils.get_logger(specs.get("path_options").get("log_dir"), scene)

            trainDataGenerator = TrainDataGenerator(specs, _logger)
            trainDataGenerator.handle_scene(scene)

            _logger.removeHandler(file_handler)
            _logger.removeHandler(stream_handler)
