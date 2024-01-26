"""
统计IBS Angle Loss 误判点的距离、角度分布，指导超参数
"""
import torch

from utils import path_utils, geometry_utils
import open3d as o3d
import os
import re
import numpy as np
import torch.nn.functional as F

from utils.test_utils import get_normalize_para


def get_geometry_path(specs, data_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, data_name).group()
    scene = re.match(scene_re, data_name).group()
    filename = re.match(filename_re, data_name).group()

    geometries_path = dict()

    pcd_dir = specs.get("path_options").get("geometries_dir").get("pcd_dir")
    Medial_axis_sphere_dir = specs.get("path_options").get("geometries_dir").get("Medial_axis_sphere_dir")

    pcd1_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_filename = '{}_{}.ply'.format(filename, 1)
    Medial_axis_sphere1_filename = '{}_0.npz'.format(scene, 0)
    Medial_axis_sphere2_filename = '{}_1.npz'.format(scene, 1)

    geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)
    geometries_path['Medial_axis_sphere1'] = os.path.join(Medial_axis_sphere_dir, category, Medial_axis_sphere1_filename)
    geometries_path['Medial_axis_sphere2'] = os.path.join(Medial_axis_sphere_dir, category, Medial_axis_sphere2_filename)

    return geometries_path


def get_angle_degree(vec1: np.ndarray, vec2: np.ndarray):
    dot_product = np.dot(vec1, vec2)

    norm_A = np.linalg.norm(vec1)
    norm_B = np.linalg.norm(vec2)

    angle_rad = np.arccos(dot_product / (norm_A * norm_B))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def visualize_intersection(geometries_path: dict):
    pcd1 = geometry_utils.read_point_cloud(geometries_path["pcd1"])
    pcd2 = geometry_utils.read_point_cloud(geometries_path["pcd2"])
    pcd1.paint_uniform_color((0.7, 0.3, 0.3))
    pcd2.paint_uniform_color((0.3, 0.7, 0.3))
    pcd1_np = np.asarray(pcd1.points)
    pcd2_np = np.asarray(pcd2.points)
    pcd1_tensor = torch.from_numpy(pcd1_np)
    pcd2_tensor = torch.from_numpy(pcd2_np)

    center1, radius1, direction1 = geometry_utils.read_medial_axis_sphere_single(geometries_path["Medial_axis_sphere1"])
    center_tensor, radius_tensor, direction1_tensor = torch.from_numpy(center1), torch.from_numpy(radius1), torch.from_numpy(direction1)

    distances = torch.cdist(pcd1_tensor, center_tensor, p=2)
    min_indices = torch.argmin(distances, dim=1)
    closest_center1 = center_tensor[min_indices, :]
    closest_direction1 = direction1_tensor[min_indices, :]
    closest_radius1 = radius_tensor[min_indices]
    min_distances = distances[torch.arange(distances.shape[0]), min_indices]
    direction_pred = pcd1_tensor - closest_center1
    cosine_sim = F.cosine_similarity(direction_pred, closest_direction1, dim=1)
    cosine_sim = torch.where(min_distances < 1.2 * closest_radius1, cosine_sim, 0)
    loss = torch.clamp(-cosine_sim - 0.5, min=0)
    interact_points_num = torch.sum(loss != 0, dim=0).float()
    if interact_points_num == 0:
        return
    else:
        print(interact_points_num)

    interact_points = (loss != 0).numpy()
    closest_center = closest_center1.numpy()
    min_indices = min_indices.numpy()
    direction_pred = direction_pred.numpy()

    arrow_pred_list = []
    arrow_prior_list = []
    for i in range(interact_points.shape[0]):
        if interact_points[i]:
            pcd1.colors[i] = (0, 0, 1)
            arrow_pred = geometry_utils.get_arrow(direction_pred[i], closest_center[i], np.linalg.norm(np.asarray(pcd1.points)[i] - closest_center[i]))
            arrow_pred.paint_uniform_color((0, 0, 1))
            arrow_pred_list.append(arrow_pred)

            arrow_prior = geometry_utils.get_arrow(direction1[min_indices[i]], center1[min_indices[i]], 0.5)
            arrow_prior.paint_uniform_color((1, 0, 0))
            arrow_prior_list.append(arrow_prior)

    o3d.visualization.draw_geometries([pcd1] + arrow_pred_list + arrow_prior_list, mesh_show_back_face=True)


def visualize(specs, filename):
    geometries_path = get_geometry_path(specs, filename)
    visualize_intersection(geometries_path)


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/ibsa_statistics.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree_dir = specs.get("path_options").get("filename_tree_dir")
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get(filename_tree_dir))

    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                print('current file: ', filename)
                try:
                    visualize(specs, filename)
                except Exception as e:
                    print(e)
