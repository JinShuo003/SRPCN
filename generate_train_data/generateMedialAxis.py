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
import json

from utils.path_utils import get_filename_tree
from utils.geometry_reader import read_point_cloud, read_mesh


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def generatePath(specs: dict, path_list: list):
    """检查specs中的path是否存在，不存在则创建"""
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

    pcd_dir = specs["pcd_dir"]
    IBS_dir = specs["IBS_dir"]

    pcd1_filename = '{}_{}.ply'.format(scene, 0)
    pcd2_filename = '{}_{}.ply'.format(scene, 1)
    IBS_filename = '{}.ply'.format(scene)

    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)
    geometries_path['IBS'] = os.path.join(IBS_dir, category, IBS_filename)

    return geometries_path


def save_mesh(specs, scene, mesh1, mesh2):
    mesh_dir = specs['mesh_normalize_save_dir']
    category = re.match(specs['category_re'], scene).group()
    # 若pcd_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    mesh1_filename = '{}_0.obj'.format(scene)
    mesh2_filename = '{}_1.obj'.format(scene)
    mesh1_path = os.path.join(mesh_dir, category, mesh1_filename)
    mesh2_path = os.path.join(mesh_dir, category, mesh2_filename)

    o3d.io.write_triangle_mesh(mesh1_path, mesh1)
    o3d.io.write_triangle_mesh(mesh2_path, mesh2)


def get_nearest_points(query, pcd):
    """
    qeury: query points
    pcd: source pcd
    return: a list contains the nearest point for every point in query in pcd
    """
    class queryPointInfo:
        def __init__(self, query, nearest, dist):
            self.query = query
            self.nearest = nearest
            self.dist = dist

    pcd_kdTree = o3d.geometry.KDTreeFlann(pcd)
    pcd_np = np.asarray(pcd.points)
    nearest_point_info = list()
    for point in np.asarray(query.points):
        [k, idx, _] = pcd_kdTree.search_knn_vector_3d(point, 1)
        dist = np.linalg.norm(point-pcd_np[idx[0]])
        nearest_point_info.append(queryPointInfo(point, pcd_np[idx[0]], dist))
    return nearest_point_info


def get_sphere_list(nearest1, nearest2):
    sphere_list = list()
    for i, info in enumerate(nearest1):
        radius = min(nearest1[i].dist, nearest2[i].dist)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere = sphere.sample_points_uniformly(64)
        sphere.paint_uniform_color((1, 0, 0))
        sphere.translate(info.query)
        sphere_list.append(sphere)
        # break
    return sphere_list


def handle_scene(specs, scene):
    """
    处理单个场景
    1. 读取完整点云和IBS点云
    2. 针对IBS点云的每个点计算球半径
    3. 保存结果
    """
    # get geometry path according scene
    geometries_path = getGeometriesPath(specs, scene)
    # get geometry
    pcd1 = read_point_cloud(geometries_path["pcd1"])
    pcd2 = read_point_cloud(geometries_path["pcd2"])
    IBS = read_point_cloud(geometries_path["IBS"])
    # calculate nearest point
    nearest1 = get_nearest_points(IBS, pcd1)
    nearest2 = get_nearest_points(IBS, pcd2)

    # visualize
    IBS.paint_uniform_color((0, 0, 0))
    pcd1.paint_uniform_color((0, 1, 0))
    pcd2.paint_uniform_color((0, 0, 1))
    sphere_list = get_sphere_list(nearest1, nearest2)
    sphere_list.append(pcd1)
    sphere_list.append(pcd2)
    sphere_list.append(IBS)

    o3d.visualization.draw_geometries(sphere_list)


def my_process(scene, specs):
    process_name = multiprocessing.current_process().name
    print(f"Running task in process: {process_name}, scene: {scene}")
    try:
        handle_scene(specs, scene)
        print(f"scene: {scene} succeed")
    except Exception as e:
        print(f"scene: {scene} failed")
        print(f"Exception info: {e.__str__()}")


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/generateMedialAxis.json'
    specs = parseConfig(config_filepath)
    # 构建文件树
    filename_tree = get_filename_tree(specs, specs["IBS_dir"])
    # 处理文件夹，不存在则创建
    generatePath(specs, ["MedialAxis_save_dir"])

    # 创建进程池，指定进程数量
    pool = multiprocessing.Pool(processes=1)
    # 参数
    scene_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            scene_list.append(scene)
    # 使用进程池执行任务，返回结果列表
    for scene in scene_list:
        pool.apply_async(my_process, (scene, specs,))

    # 关闭进程池
    pool.close()
    pool.join()
