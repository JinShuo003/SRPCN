"""
从成对的Mesh采集具有遮挡关系的残缺点云
"""

import sys

import utils

sys.path.append("/home/data/jinshuo/IBPCDC")
import math
import os
import re
import multiprocessing
import open3d as o3d
import numpy as np
import json
from utils import path_utils, geometry_utils


class SampleMethodException(Exception):
    def __init__(self, message="Illegal sample method, surface or IOU are supported"):
        self.message = message
        super.__init__(message)


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def generatePath(specs: dict, path_list: list):
    for path in path_list:
        if not os.path.isdir(specs[path]):
            os.makedirs(specs[path])


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)


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


class BorderNotSetException(Exception):
    def __init__(self):
        self.msg = "Border haven't been set"


class DirectionNotSetException(Exception):
    def __init__(self):
        self.msg = "Direction is None, you should set border first, the direction will be computed automatically"


class DataTypeInvalidException(Exception):
    def __init__(self):
        self.msg = "The type of the point should be np.ndarray"


class DataDemensionInvalidException(Exception):
    def __init__(self):
        self.msg = "The demension of the point should be (1, 3)"


class PlaneInfo:
    def __init__(self):
        # 平面的四个角点，左上、左下、右上、右下
        self.border: tuple = None
        # 平面的四个方向向量，左、右、上、下
        self.direction: tuple = None

    def set_border(self, border: tuple):
        if border.__len__() != 4:
            raise Exception("border should have 4 points")
        for point in border:
            if not isinstance(point, np.ndarray):
                raise DataTypeInvalidException()
            if point.shape != (1, 3):
                raise DataDemensionInvalidException()
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
    """按照specs中的配置对mesh1和mesh2进行单角度扫描，得到若干个视角的单角度残缺点云"""

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
        self.scan_plane = PlaneInfo()
        self.width_px = self.specs["scan_options"]["width_px"]-1
        self.height_px = self.specs["scan_options"]["height_px"]-1
        self.step_length_row = 0
        self.step_length_col = 0

    def get_ray_casting_scene(self):
        """初始化光线追踪场景"""
        try:
            mesh1_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh1)
            mesh2_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh2)
            self.scene.add_triangles(mesh1_t)
            self.scene.add_triangles(mesh2_t)
        except Exception as e:
            print(e)

    def get_step_length(self):
        self.step_length_row = np.linalg.norm(self.scan_plane.get_left_up() - self.scan_plane.get_right_up()) / self.width_px
        self.step_length_col = np.linalg.norm(self.scan_plane.get_left_up() - self.scan_plane.get_left_down()) / self.height_px

    def update_plane(self, rays):
        # 根据光线构造虚平面
        self.scan_plane.set_border(self.get_border_points(rays))
        # 获取横、纵方向上的步长
        self.get_step_length()

    def get_init_rays(self, theta, camera_height, r, fov_deg):
        # 获取初始光线
        eye = [r * math.cos(theta), camera_height, r * math.sin(theta)]
        rays = self.scene.create_rays_pinhole(fov_deg=fov_deg,
                                              center=[0, 0, 0],
                                              eye=eye,
                                              up=[0, 1, 0],
                                              width_px=8,
                                              height_px=8)
        self.update_plane(rays)

        # 求出虚平面上所有光线的投影点
        points = []
        # 第一行
        row0_points = []
        for i in range(self.width_px+1):
            point = self.scan_plane.get_left_up() + i*self.step_length_row*self.scan_plane.get_dir_right()
            row0_points.append(point)
        # 将第一行向下扩展
        for point in row0_points:
            for i in range(self.height_px+1):
                point_ = point + i * self.step_length_col * self.scan_plane.get_dir_down()
                points.append(point_)

        # points_ = np.array(points).reshape(256, 3)
        # eye = np.array(eye).reshape(1, 3)
        # points_ = np.concatenate((eye, points_), axis=0)
        # lines = [[0, i] for i in range(1, points_.shape[0]-1)]
        # colors = [[1, 0, 0] for i in range(lines.__len__())]
        # lines_pcd = o3d.geometry.LineSet()
        # lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        # lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
        # lines_pcd.points = o3d.utility.Vector3dVector(points_)
        # mesh_sphere = geometry_utils.get_sphere_pcd()
        # o3d.visualization.draw_geometries([lines_pcd, mesh_sphere, self.mesh1, self.mesh2])

        # 构造open3d ray，(eye, direction): Tensor(n)
        rays = []
        eye = np.array(eye).reshape(1, 3)
        for point in points:
            direction = (point-eye) / np.linalg.norm((point-eye))
            rays.append(np.concatenate((eye, direction), axis=1).reshape(6))
        return o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)

    def get_border_points(self, rays):
        rays_ = rays.numpy()[:, :, 3:6]
        row, col, _ = rays_.shape
        left_up = rays_[row - 1][col - 1].reshape(1, 3)
        left_down = rays_[0][col - 1].reshape(1, 3)
        right_up = rays_[row - 1][0].reshape(1, 3)
        right_down = rays_[0][0].reshape(1, 3)
        return left_up, left_down, right_up, right_down

    def get_rays(self, theta, camera_height, r, fov_deg):
        """获取当前视角的光线"""
        eye = [r * math.cos(theta), camera_height, r * math.sin(theta)]
        rays = self.scene.create_rays_pinhole(fov_deg=fov_deg,
                                              center=[0, 0, 0],
                                              eye=eye,
                                              up=[0, 1, 0],
                                              width_px=self.specs["scan_options"]["width_px"],
                                              height_px=self.specs["scan_options"]["height_px"])
        return rays

    def visualize_rays(self, rays):
        # 可视化光线和mesh
        if self.visualize:
            cast_rays = self.get_rays_visualization(rays)
            cast_rays_single = self.get_rays_visualization_single_view(rays)
            mesh_sphere = geometry_utils.get_sphere_pcd()
            o3d.visualization.draw_geometries([cast_rays, cast_rays_single, mesh_sphere, self.mesh1, self.mesh2])

    def get_ray_cast_result(self, rays):
        return self.scene.cast_rays(rays)

    def should_iterate_continue(self, cast_result):
        """
        判断当前光线求交的结果是否满足分辨率要求，既和物体1、物体2相交的光线数量是否足够
        """
        geometry_ids = cast_result["geometry_ids"].numpy()
        points_num_expect = self.specs["scan_options"]["points_num"]
        points_num_geometry0 = 0
        points_num_geometry1 = 0
        for id in geometry_ids:
            if id == 0:
                points_num_geometry0 += 1
            if id == 1:
                points_num_geometry1 += 1
        return points_num_geometry0 >= points_num_expect and points_num_geometry1 >= points_num_expect

    def generate_scan_pcd(self):
        scan_options = self.specs["scan_options"]
        view_num = scan_options["view_num"]
        camera_height = scan_options["camera_height"]
        camera_ridius = scan_options["camera_ridius"]
        fov_deg = scan_options["fov_deg"]

        pcd1_partial_list = []
        pcd2_partial_list = []
        scan_view_list = []
        for i in range(scan_options["view_num"]):
            # get init seed rays
            rays = self.get_init_rays(theta=2 * math.pi * i / view_num, camera_height=camera_height, r=camera_ridius, fov_deg=fov_deg)
            # self.visualize_rays(init_rays)
            cast_result = self.get_ray_cast_result(rays)
            rays_hit_object0, rays_hit_object1 = self.count_hit_num()
            # iterate while the points number of two objects not enough
            while not self.should_iterate_continue(cast_result):
                # extend the rays according to the cast result
                rays = self.expand_rays(rays, cast_result)
                cast_result = self.get_ray_cast_result(rays)
            # # 根据光线投射结果获取当前角度的残缺点云
            # pcd1_scan, pcd2_scan = self.get_cur_view_pcd(init_rays, cast_result)
            # # 两者都采样成功则保存，并记录视角下标，否则丢弃
            # if pcd1_scan and pcd2_scan:
            #     pcd1_partial_list.append(pcd1_scan)
            #     pcd2_partial_list.append(pcd2_scan)
            #     scan_view_list.append(i)
            # else:
            #     print('view: {}, vertices not enough, sample failed'.format(i))

        return pcd1_partial_list, pcd2_partial_list, scan_view_list

    def get_cur_view_pcd(self, rays, cast_result):
        hit = cast_result['t_hit'].numpy()
        geometry_ids = cast_result["geometry_ids"].numpy()
        primitive_ids = cast_result["primitive_ids"].numpy()
        primitive_uvs = cast_result["primitive_uvs"].numpy()

        points_pcd1 = []
        points_pcd2 = []

        # 获取光线击中的点
        for i in range(rays.shape[0]):
            for j in range(rays.shape[1]):
                if not math.isinf(hit[i][j]):
                    if geometry_ids[i][j] == 0:
                        points_pcd1.append(
                            self.get_real_coordinate(self.mesh1_vertices, self.mesh1_triangels[primitive_ids[i][j]],
                                                     primitive_uvs[i][j]))
                    if geometry_ids[i][j] == 1:
                        points_pcd2.append(
                            self.get_real_coordinate(self.mesh2_vertices, self.mesh2_triangels[primitive_ids[i][j]],
                                                     primitive_uvs[i][j]))

        pcd_sample_options = self.specs["scan_options"]
        pcd1_scan = o3d.geometry.PointCloud()
        pcd2_scan = o3d.geometry.PointCloud()
        pcd1_scan.points = o3d.utility.Vector3dVector(points_pcd1)
        pcd2_scan.points = o3d.utility.Vector3dVector(points_pcd2)
        try:
            pcd1_scan = pcd1_scan.farthest_point_down_sample(pcd_sample_options["points_num"])
            pcd2_scan = pcd2_scan.farthest_point_down_sample(pcd_sample_options["points_num"])
            pcd1_scan.paint_uniform_color((1, 0, 0))
            pcd2_scan.paint_uniform_color((0, 1, 0))
        except:
            pcd1_scan = None
            pcd2_scan = None

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
            points.append(rays_[i]+eye)
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
        sphere = geometry_utils.get_unit_sphere_pcd()

        for i in range(len(pcd1_partial_list)):
            o3d.visualization.draw_geometries(
                [pcd1_partial_list[i], pcd2_partial_list[i], coor, sphere],
                window_name="{}".format(scan_view_list[i]))

    def handle_scene(self, scene):
        self.geometries_path = getGeometriesPath(self.specs, scene)
        self.get_init_geometries()

        # generate single view scan point cloud
        print("begin generate scan pointcloud")
        pcd1_partial_list, pcd2_partial_list, scan_view_list = self.get_scan_pcd()

        if self.specs["visualize"]:
            self.visualize_result(None, None, pcd1_partial_list, pcd2_partial_list,
                                  scan_view_list)
        # save the result
        save_pcd(self.specs, pcd1_partial_list, pcd2_partial_list, scan_view_list, scene)


def my_process(scene, specs):
    process_name = multiprocessing.current_process().name
    print(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs)

    try:
        trainDataGenerator.handle_scene(scene)
        print("scene: {} succeed".format(scene))
    except Exception as e:
        print("scene: {} failed, exception message: {}".format(scene, e.message))


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/generateTrainData.json'
    specs = parseConfig(config_filepath)
    processNum = specs["process_num"]
    # 构建文件树
    filename_tree = path_utils.getFilenameTree(specs, specs["mesh_dir"])
    # 处理文件夹，不存在则创建
    generatePath(specs, ["pcd_partial_save_dir"])

    # 创建进程池，指定进程数量
    # pool = multiprocessing.Pool(processes=processNum)

    scene_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            scene_list.append(scene)
    trainDataGenerator = TrainDataGenerator(specs)
    # 使用进程池执行任务，返回结果列表
    for scene in scene_list:
        print("current scene: {}".format(scene))
        # pool.apply_async(my_process, (scene, specs,))
        trainDataGenerator.handle_scene(scene)

    # # 关闭进程池
    # pool.close()
    # pool.join()
