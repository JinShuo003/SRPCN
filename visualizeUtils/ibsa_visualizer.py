"""
统计IBS Angle Loss 误判点的距离、角度分布，指导超参数
"""
import torch

from utils import path_utils, geometry_utils, random_utils
import open3d as o3d
import os
import re
import numpy as np
import torch.nn.functional as F

intersect_num_category = 0
data_len_category = 0
intersect_num_total = 0
data_len_total = 0


def get_geometry_path(specs, data_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, data_name).group()
    scene = re.match(scene_re, data_name).group()
    filename = re.match(filename_re, data_name).group()

    geometries_path = dict()

    mesh_dir = specs.get("path_options").get("geometries_dir").get("mesh_dir")
    pcd_dir = specs.get("path_options").get("geometries_dir").get("pcd_dir")
    IBS_mesh_dir = specs.get("path_options").get("geometries_dir").get("IBS_mesh_dir")
    Medial_axis_sphere_dir = specs.get("path_options").get("geometries_dir").get("Medial_axis_sphere_dir")
    Normalize_data_dir = specs.get("path_options").get("geometries_dir").get("Normalize_data_dir")

    mesh1_filename = '{}_{}.obj'.format(scene, 0)
    mesh2_filename = '{}_{}.obj'.format(scene, 1)
    pcd1_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_filename = '{}_{}.ply'.format(filename, 1)
    IBS_mesh_filename = '{}.obj'.format(scene)
    Medial_axis_sphere_filename = '{}.npz'.format(scene)
    Medial_axis_sphere1_filename = '{}_0.npz'.format(scene, 0)
    Medial_axis_sphere2_filename = '{}_1.npz'.format(scene, 1)
    normalize_data1_filename = '{}_0.txt'.format(scene, 0)
    normalize_data2_filename = '{}_1.txt'.format(scene, 1)

    geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)
    geometries_path['IBS_mesh'] = os.path.join(IBS_mesh_dir, category, IBS_mesh_filename)
    geometries_path['Medial_axis_sphere'] = os.path.join(Medial_axis_sphere_dir, category,
                                                         Medial_axis_sphere_filename)
    geometries_path['Medial_axis_sphere1'] = os.path.join(Medial_axis_sphere_dir, category,
                                                          Medial_axis_sphere1_filename)
    geometries_path['Medial_axis_sphere2'] = os.path.join(Medial_axis_sphere_dir, category,
                                                          Medial_axis_sphere2_filename)
    geometries_path['normalize_data1'] = os.path.join(Normalize_data_dir, category, normalize_data1_filename)
    geometries_path['normalize_data2'] = os.path.join(Normalize_data_dir, category, normalize_data2_filename)

    return geometries_path


def get_angle_degree(vec1: np.ndarray, vec2: np.ndarray):
    dot_product = np.dot(vec1, vec2)

    norm_A = np.linalg.norm(vec1)
    norm_B = np.linalg.norm(vec2)

    angle_rad = np.arccos(dot_product / (norm_A * norm_B))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_intersection(pcd: torch.Tensor, center: torch.Tensor, radius: torch.Tensor, direction: torch.Tensor,
                         angle_threshold: int = 90, distance_ratio_threhold: float = 1):
    distances = torch.cdist(pcd, center, p=2)
    min_indices = torch.argmin(distances, dim=1)
    closest_center = center[min_indices, :]
    closest_direction = direction[min_indices, :]
    closest_radius = radius[min_indices]
    min_distances = distances[torch.arange(distances.shape[0]), min_indices]
    direction_pred = pcd - closest_center
    direction_pred /= torch.norm(direction_pred, dim=1, keepdim=True)
    cosine_sim = -F.cosine_similarity(direction_pred, closest_direction, dim=1) + np.cos(np.deg2rad(angle_threshold))
    cosine_sim = torch.clamp(cosine_sim, min=0)
    # cosine_sim = torch.where(min_distances < distance_ratio_threhold * closest_radius, cosine_sim, 0)
    intersect_points_num = torch.sum(cosine_sim != 0, dim=0).int().item()
    if intersect_points_num != 0:
        global intersect_num_category
        intersect_num_category += intersect_points_num
        print("intersect_points_num: {}".format(intersect_points_num))

    return closest_center, closest_direction, closest_radius, direction_pred, cosine_sim, intersect_points_num


def visualize(pcd, IBS, cosine_sim, closest_center, closest_direction, direction_pred, show_positive=True, geometry_list=[]):
    pcd_color = (1, 0, 0) if show_positive is True else (0, 0, 1)
    arrow_pred_color = (1, 0, 0) if show_positive is True else (0, 0, 1)
    arrow_prior_color = (0, 1, 0)
    pcd_np = np.asarray(pcd.points)
    interact_points = (cosine_sim != 0).numpy()
    closest_center = closest_center.numpy()
    closest_direction = closest_direction.numpy()
    direction_pred = direction_pred.numpy()
    arrow_pred_list = []
    arrow_prior_list = []
    for i in range(interact_points.shape[0]):
        angle = get_angle_degree(closest_direction[i], direction_pred[i])
        if not interact_points[i] ^ show_positive:
            # print(angle)
            pcd.colors[i] = pcd_color
            arrow_pred = geometry_utils.get_arrow(direction_pred[i], closest_center[i],
                                                  np.linalg.norm(pcd_np[i] - closest_center[i]))
            arrow_pred.paint_uniform_color(arrow_pred_color)
            arrow_pred_list.append(arrow_pred)

            arrow_prior = geometry_utils.get_arrow(closest_direction[i], closest_center[i], 0.2)
            arrow_prior.paint_uniform_color(arrow_prior_color)
            arrow_prior_list.append(arrow_prior)

    o3d.visualization.draw_geometries([pcd, IBS] + geometry_list, mesh_show_back_face=True, mesh_show_wireframe=True)
    # o3d.visualization.draw_geometries([pcd, IBS] + arrow_pred_list + arrow_prior_list, mesh_show_back_face=True)


def handle_data_FP(geometries_path: dict, tag: str):
    pcd = geometry_utils.read_point_cloud(geometries_path["pcd{}".format(tag)])
    pcd.paint_uniform_color((0, 0, 0.7))
    pcd_np = np.asarray(pcd.points)
    pcd_tensor = torch.from_numpy(pcd_np)

    IBS_mesh = geometry_utils.read_mesh(geometries_path["IBS_mesh"])
    IBS_mesh.paint_uniform_color((0.7, 0.7, 0.3))

    center, radius, direction1, direction2 = geometry_utils.read_medial_axis_sphere_total(
        geometries_path["Medial_axis_sphere"])
    direction = direction1 if tag == "1" else direction2
    center_tensor, radius_tensor, direction_tensor = torch.from_numpy(center), torch.from_numpy(
        radius), torch.from_numpy(direction)

    closest_center, closest_direction, closest_radius, direction_pred, cosine_sim, intersect_points_num = compute_intersection(
        pcd_tensor, center_tensor, radius_tensor, direction_tensor, 150, 1.5
    )
    if intersect_points_num != 0:
        global intersect_num_category
        intersect_num_category += intersect_points_num
        print("intersect_points_num: {}".format(intersect_points_num))

    global data_len_category
    data_len_category += 1

    if specs.get("visualize") and intersect_points_num:
        visualize(pcd, IBS_mesh, cosine_sim, closest_center, closest_direction, direction_pred)


def handle_data_FN(geometries_path: dict, tag: str):
    mesh = geometry_utils.read_mesh(geometries_path["mesh{}".format(tag)])
    mesh.paint_uniform_color((0, 0.5, 0))

    pcd = geometry_utils.read_point_cloud(geometries_path["pcd{}".format(tag)])
    pcd_aabb = pcd.get_axis_aligned_bounding_box().scale(1.5, pcd.get_center())
    pcd.paint_uniform_color((0, 1, 0))
    pcd_np = np.asarray(pcd.points)
    pcd_tensor = torch.from_numpy(pcd_np)

    IBS_mesh = geometry_utils.read_mesh(geometries_path["IBS_mesh"])
    IBS_mesh.paint_uniform_color((0.7, 0.7, 0.3))
    IBS_aabb = IBS_mesh.get_axis_aligned_bounding_box().scale(1.2, IBS_mesh.get_center())

    test_points_IBS_np = random_utils.get_random_points_in_aabb(IBS_aabb, 1024*4)
    test_points_pcd_np = random_utils.get_random_points_in_aabb(pcd_aabb, 1024*4)
    test_points_np = np.concatenate((test_points_IBS_np, test_points_pcd_np), axis=0)
    # test_points_np = test_points_IBS_np
    test_points_pcd = geometry_utils.get_pcd_from_np(test_points_np)
    test_points_pcd.paint_uniform_color((1, 0, 0))
    test_points_tensor = torch.from_numpy(test_points_np)

    center, radius, direction1, direction2 = geometry_utils.read_medial_axis_sphere_total(
        geometries_path["Medial_axis_sphere"])
    direction = direction1 if tag == "1" else direction2
    center_tensor, radius_tensor, direction_tensor = torch.from_numpy(center), torch.from_numpy(
        radius), torch.from_numpy(direction)

    closest_center, closest_direction, closest_radius, direction_pred, cosine_sim, intersect_points_num = compute_intersection(
        test_points_tensor, center_tensor, radius_tensor, direction_tensor, 90, 2
    )

    global data_len_category
    data_len_category += 1

    if specs.get("visualize") and intersect_points_num != 2048:
        visualize(test_points_pcd, IBS_mesh, cosine_sim, closest_center, closest_direction, direction_pred,
                  False, [])


def handle_data(specs, filename):
    geometries_path = get_geometry_path(specs, filename)
    """
    阳性：某个点被ibsa指标判别为出现在错误一侧，需要参与loss计算
    阴性：某个点被ibsa指标判别为出现在正确一侧，不需要参与loss计算
    FP: False Positive，指应当判别为阴性，但错误地判别为了阳性，比如一个点云均在ibs正确的一侧，但是某些点被指标判别为了错误一侧
    FN: False Negative，指应当判别为阳性，但错误地判别为了阴性，比如一个点云均在ibs错误的一侧，但是某些点被指标判别为了错误一侧
    """
    if specs.get("mode") == "FP":
        handle_data_FP(geometries_path, "1")
        handle_data_FP(geometries_path, "2")
    elif specs.get("mode") == "FN":
        handle_data_FN(geometries_path, "1")
        handle_data_FN(geometries_path, "2")


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/ibsa_visualizer.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree_dir = specs.get("path_options").get("filename_tree_dir")
    filename_tree = path_utils.get_filename_tree(specs,
                                                 specs.get("path_options").get("geometries_dir").get(filename_tree_dir))

    for category in filename_tree:
        print('current category: ', category)
        for scene in filename_tree[category]:
            print('current scene: ', scene)
            for filename in filename_tree[category][scene]:
                try:
                    handle_data(specs, filename)
                except Exception as e:
                    print(e)
        print(intersect_num_category)
        print(data_len_category)
        print(intersect_num_category / (data_len_category * 2048))
        intersect_num_category = 0
        data_len_category = 0

    # print(intersect_num_category)
    # print(data_len_category)
    # print(intersect_num_category / (data_len_category * 2048))
