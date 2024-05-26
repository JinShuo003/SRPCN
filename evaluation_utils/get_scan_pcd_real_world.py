"""
从真实场景的若干个Mesh中采集具有遮挡关系的残缺点云
"""
import logging
import math
import os
import random
import re

import numpy as np
import open3d as o3d

from utils import path_utils, geometry_utils, random_utils, exception_utils, log_utils

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


def read_mesh(mesh_filename_list: list):
    mesh_list = []
    for filename in mesh_filename_list:
        mesh = geometry_utils.read_mesh(filename)
        mesh_list.append(mesh)
    return mesh_list


def get_scan_pcd(scan_options, mesh_list: list):
    mesh_num = len(mesh_list)
    scan_pcd_list = [[] for i in range(mesh_num)]
    scan_view_list = []
    # 球坐标，theta为天顶角，phi为方位角
    index = 0
    logging.info("begin generate theta: {}, phi: {}".format(0, 0))
    pcd_scan, success = get_current_view_scan_pcd(scan_options, mesh_list, 0, 0)
    if success:
        for i, pcd in enumerate(pcd_scan):
            scan_pcd_list[i].append(pcd)
        scan_view_list.append(index)
        index += 1
    logging.info("begin generate theta: {}, phi: {}".format(180, 0))
    pcd_scan, success = get_current_view_scan_pcd(scan_options, mesh_list, 180, 0)
    if success:
        for i, pcd in enumerate(pcd_scan):
            scan_pcd_list[i].append(pcd)
        scan_view_list.append(index)
        index += 1
    for theta in [45, 90, 135]:
        for phi in range(0, 360, 45):
            logging.info("begin generate theta: {}, phi: {}".format(theta, phi))
            pcd_scan, success = get_current_view_scan_pcd(scan_options, mesh_list, theta, phi)
            if success:
                for i, pcd in enumerate(pcd_scan):
                    scan_pcd_list[i].append(pcd)
                scan_view_list.append(index)
                index += 1

    return scan_pcd_list, scan_view_list


def get_view_point(theta, phi, r):
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


def get_border_points(eye, rays):
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


def build_plane(eye, rays):
    """根据光线构造虚平面"""
    plane = Plane()
    plane.set_border(get_border_points(eye, rays))
    return plane


def get_projection_plane(scene, eye, fov_deg):
    """
    根据视点和视场角获取虚拟投影平面
    Args:
        eye: 视点
        fov_deg: 视场角
    Returns:
        投影平面
    """
    # 视点朝向(0, 0, 0)，发射8*8条光线
    rays = scene.create_rays_pinhole(fov_deg=fov_deg,
                                     center=[0, 0, 0],
                                     eye=eye,
                                     up=[0, 1, 0],
                                     width_px=8,
                                     height_px=8)
    return build_plane(eye, rays)


def get_pixel_size(plane: Plane, widthResolution: int, heightResolution: int):
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


def get_projection_points(projection_plane: Plane, resolution_width: int, resolution_height: int):
    """
    给定投影平面、宽高方向上的分辨率，计算平面上的所有投影点
    Args:
        projection_plane: 投影平面
        resolution_width: 宽方向上的分辨率，即投影平面的宽方向上期望有多少个点
        resolution_height: 高方向上的分辨率，即投影平面的高方向上期望有多少个点
    Returns:
        投影点，type: np.ndarray，shape: (n, 3)
    """
    pixel_width, pixel_height = get_pixel_size(projection_plane, resolution_width, resolution_height)
    projection_points = []
    for i in range(resolution_width):
        for j in range(resolution_height):
            point = projection_plane.get_left_up() + \
                    i * pixel_width * projection_plane.get_dir_right() + \
                    j * pixel_height * projection_plane.get_dir_down()
            projection_points.append(point)

    projection_points = np.array(projection_points).reshape(-1, 3)
    return pixel_width, pixel_height, projection_points


def expand_points_in_rectangle(expand_points_num, width, height, plane: Plane, points):
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


def get_rays_from_projection_points(eye, projection_points):
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


def get_ray_cast_result(scene, rays):
    """获取光线投射结果"""
    return scene.cast_rays(rays)


def get_points_intersect(mesh_num: int, projection_points: np.ndarray, cast_result):
    """
    从光线在投影平面的投影点中筛选出和和每个物体相交的部分
    Args:
        projection_points: 所有投影点
        cast_result: 射线求交的结果
    Returns:
        与每个mesh相交的点
    """
    assert projection_points.shape[0] == cast_result["t_hit"].shape[0]

    geometry_ids = cast_result["geometry_ids"].numpy()
    points_intersect_total = [[] for _ in range(mesh_num)]
    for ray_id, intersect_id in enumerate(geometry_ids):
        if intersect_id < mesh_num:
            points_intersect_total[intersect_id].append(projection_points[ray_id])

    points_intersect_result = []
    for points_intersect in points_intersect_total:
        points_intersect_result.append(np.array(points_intersect).reshape(-1, 3))
    return points_intersect_result


def is_view_legal(points_intersect_list: list):
    """
    判定当前射线求交结果是否能够满足采集数据的要求
    Args:
        points_intersect_list: 与物体相交的射线投影点列表
    Returns:
        是否满足采集条件
    """
    for points_intersect in points_intersect_list:
        if points_intersect.shape[0] == 0:
            return False
    return True


def get_real_coordinate(vertices, triangles, uv_coordinate):
    # 将三角形的重心坐标变换为真实坐标
    point1 = vertices[triangles[0]]
    point2 = vertices[triangles[1]]
    point3 = vertices[triangles[2]]
    return uv_coordinate[0] * point1 + uv_coordinate[1] * point2 + (
            1 - uv_coordinate[0] - uv_coordinate[1]) * point3


def get_cur_view_pcd(mesh_list: list, mesh_num: int, cast_result):
    mesh_vertices = []
    mesh_triangles = []
    for i in range(mesh_num):
        mesh_vertices.append(np.asarray(mesh_list[i].vertices))
        mesh_triangles.append(np.asarray(mesh_list[i].triangles))

    hit = cast_result['t_hit'].numpy()
    geometry_ids = cast_result["geometry_ids"].numpy()
    primitive_ids = cast_result["primitive_ids"].numpy()
    primitive_uvs = cast_result["primitive_uvs"].numpy()

    points_list = [[] for i in range(mesh_num)]

    # get points from ray casting result
    for i in range(hit.shape[0]):
        if not math.isinf(hit[i]):
            geometry_id = geometry_ids[i]
            points_list[geometry_id].append(get_real_coordinate(mesh_vertices[geometry_id],
                                                                mesh_triangles[geometry_id][primitive_ids[i]],
                                                                primitive_uvs[i]))

    pcd_list = []
    for i in range(mesh_num):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_list[i])
        pcd_list.append(pcd)

    return pcd_list


def get_ray_casting_scene(mesh_list: list):
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in mesh_list:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
    return scene


def visualize_rays(eye, rays, geometries):
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

    o3d.visualization.draw_geometries(geometries + [lines_pcd], mesh_show_wireframe=True)


def is_points_enough(points_intersect_list: list, pcd_sample_num: int):
    for points_intersect in points_intersect_list:
        if points_intersect.shape[0] < pcd_sample_num:
            return False
    return True


def concatenate_points(points_list: list):
    points_result = None
    for points in points_list:
        points_result = points if points_result is None else np.concatenate((points_result, points))
    return points_result


def concatenate_rays(rays_list: list):
    rays_result = None
    for rays in rays_list:
        rays = rays.numpy()
        rays_result = rays if rays_result is None else np.concatenate((rays_result, rays))
    return rays_result


def get_current_view_scan_pcd(scan_options: dict, mesh_list: list, theta: float, phi: float):
    """
    获取某个角度观察的残缺点云数据
    Args:
        theta: 球坐标天顶角
        phi: 球坐标方位角
    Returns:
        残缺点云列表，是否采集成功
    """
    mesh_num = len(mesh_list)
    # add each mesh to ray casting scene
    scene = get_ray_casting_scene(mesh_list)

    camera_ridius = scan_options["camera_ridius"]
    fov_deg = scan_options["fov_deg"]
    pcd_point_num = scan_options["points_num"]
    expand_points_num = scan_options["expand_points_num"]
    pcd_sample_num = 1.5 * pcd_point_num
    assert pcd_sample_num > pcd_point_num
    resolution_width = scan_options["resolution_width"]
    resolution_height = scan_options["resolution_height"]

    # 按照配置分辨率获取初始光线的相关信息
    eye = get_view_point(theta, phi, camera_ridius)  # viewport，(3)
    scan_plane = get_projection_plane(scene=scene, eye=eye, fov_deg=fov_deg)  # projection plane
    pixel_width, pixel_height, projection_points = get_projection_points(scan_plane, resolution_width,
                                                                         resolution_height)  # 投影点
    projection_points = expand_points_in_rectangle(expand_points_num, pixel_width, pixel_height, scan_plane,
                                                   projection_points)  # 扩充投影点，保证随机性
    rays = get_rays_from_projection_points(eye, projection_points)  # 射线
    # visualize_rays(eye, rays, mesh_list)
    cast_result = get_ray_cast_result(scene, rays)  # 射线求交结果
    points_intersect_list = get_points_intersect(mesh_num, projection_points, cast_result)  # 与obj1、obj2相交的射线投影点
    # 判断初始结果是否满足采集条件
    if not is_view_legal(points_intersect_list):
        logging.warning("not enough init points, theta: {}, phi: {}".format(theta, phi))
        return None, False
    rays_list = []
    for i in range(mesh_num):
        rays_list.append(get_rays_from_projection_points(eye, points_intersect_list[i]))
        # visualize_rays(eye, rays_obj[i], mesh_list)

    # iterate when any mesh don't have enough points
    while not is_points_enough(points_intersect_list, pcd_sample_num):
        for i in range(mesh_num):
            if points_intersect_list[i].shape[0] < pcd_sample_num:
                logging.info("intersect points with obj{} not enough, cur: {}, target: {}"
                             .format(i, points_intersect_list[i].shape[0], pcd_sample_num))
                points = expand_points_in_rectangle(expand_points_num,
                                                    pixel_width,
                                                    pixel_height,
                                                    scan_plane,
                                                    points_intersect_list[i])
                rays = get_rays_from_projection_points(eye, points)
                points_intersect_list[i] = points
                rays_list[i] = rays
        # concatenate all the rays and do rays cast again
        # concatenate projection points
        projection_points = concatenate_points(points_intersect_list)
        # concatenate rays
        rays = concatenate_rays(rays_list)
        rays = o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)
        cast_result = get_ray_cast_result(scene, rays)
        points_intersect_list = get_points_intersect(mesh_num, projection_points, cast_result)
        rays_list = []
        for i in range(mesh_num):
            rays_list.append(get_rays_from_projection_points(eye, points_intersect_list[i]))

    # get partial point cloud from ray casting result
    pcd_list = get_cur_view_pcd(mesh_list, mesh_num, cast_result)
    for i, pcd in enumerate(pcd_list):
        pcd_list[i] = pcd.farthest_point_down_sample(pcd_point_num)
        pcd_list[i].paint_uniform_color((random.random(), random.random(), random.random()))

    # o3d.visualization.draw_geometries(pcd_list)
    return pcd_list, True


def save_scan_pcd(pcd_partial_dir: str, mesh_filename_list: list, scan_pcd_list: list, scan_view_list: list):
    mesh_num = len(mesh_filename_list)
    view_num = len(scan_view_list)
    for i in range(mesh_num):
        filename = os.path.basename(mesh_filename_list[i])
        filename, extension = os.path.splitext(filename)
        category = re.match('scene\\d', filename).group()
        scene_name = re.match('scene\\d.\\d{4}', filename).group()
        idx = filename[-1]
        for j in range(view_num):
            pcd_save_path = os.path.join(pcd_partial_dir, category)
            path_utils.generate_path(pcd_save_path)
            pcd_save_path = os.path.join(pcd_save_path, '{}_view{}_{}.ply'.format(scene_name, scan_view_list[j], idx))
            o3d.io.write_point_cloud(pcd_save_path, scan_pcd_list[i][j])


if __name__ == '__main__':
    mesh_filename_list = [
        r'D:\dataset\IBPCDC\real-world\mesh\scene1\scene1.1012_0.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene1\scene1.1012_1.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene2\scene2.1000_0.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene2\scene2.1000_1.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene3\scene3.1007_0.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene3\scene3.1007_1.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene9\scene9.1001_0.obj',
        r'D:\dataset\IBPCDC\real-world\mesh\scene9\scene9.1001_1.obj'
    ]

    scan_options = {
        "camera_ridius": 1,
        "fov_deg": 70,
        "resolution_width": 256,
        "resolution_height": 256,
        "min_init_point_num": 30,
        "min_init_radius": 0.01,
        "points_num": 2048,
        "expand_points_num": 5
    }

    # read mesh
    mesh_list = read_mesh(mesh_filename_list)
    # o3d.visualization.draw_geometries(mesh_list)

    # get scan pcd
    scan_pcd_list, scan_view_list = get_scan_pcd(scan_options, mesh_list)

    # save scan pcd
    pcd_partial_dir = r'D:\dataset\IBPCDC\real-world\pcdScan\INTE'
    save_scan_pcd(pcd_partial_dir, mesh_filename_list, scan_pcd_list, scan_view_list)
