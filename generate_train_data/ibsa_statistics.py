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

intersect_num_total = 0
data_len = 0


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
    geometries_path['Medial_axis_sphere1'] = os.path.join(Medial_axis_sphere_dir, category,
                                                          Medial_axis_sphere1_filename)
    geometries_path['Medial_axis_sphere2'] = os.path.join(Medial_axis_sphere_dir, category,
                                                          Medial_axis_sphere2_filename)

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
    center2, radius2, direction2 = geometry_utils.read_medial_axis_sphere_single(geometries_path["Medial_axis_sphere2"])
    center_tensor1, radius_tensor1, direction1_tensor = torch.from_numpy(center1), torch.from_numpy(
        radius1), torch.from_numpy(direction1)
    center_tensor2, radius_tensor2, direction2_tensor = torch.from_numpy(center2), torch.from_numpy(
        radius2), torch.from_numpy(direction2)

    distances1 = torch.cdist(pcd1_tensor, center_tensor1, p=2)
    distances2 = torch.cdist(pcd2_tensor, center_tensor2, p=2)
    min_indices1 = torch.argmin(distances1, dim=1)
    min_indices2 = torch.argmin(distances2, dim=1)
    closest_center1 = center_tensor1[min_indices1, :]
    closest_center2 = center_tensor2[min_indices2, :]
    closest_direction1 = direction1_tensor[min_indices1, :]
    closest_direction2 = direction2_tensor[min_indices2, :]
    closest_radius1 = radius_tensor1[min_indices1]
    closest_radius2 = radius_tensor2[min_indices2]
    min_distances1 = distances1[torch.arange(distances1.shape[0]), min_indices1]
    min_distances2 = distances2[torch.arange(distances2.shape[0]), min_indices2]
    direction_pred1 = pcd1_tensor - closest_center1
    direction_pred2 = pcd2_tensor - closest_center2
    direction_pred1 /= torch.norm(direction_pred1, dim=1, keepdim=True)
    direction_pred2 /= torch.norm(direction_pred2, dim=1, keepdim=True)
    cosine_sim1 = -F.cosine_similarity(direction_pred1, closest_direction1, dim=1) + np.cos(np.deg2rad(120))
    cosine_sim2 = -F.cosine_similarity(direction_pred2, closest_direction2, dim=1) + np.cos(np.deg2rad(120))
    cosine_sim1 = torch.where(min_distances1 < 1.5 * closest_radius1, cosine_sim1, 0)
    cosine_sim2 = torch.where(min_distances2 < 1.5 * closest_radius2, cosine_sim2, 0)
    cosine_sim1 = torch.clamp(cosine_sim1, min=0)
    cosine_sim2 = torch.clamp(cosine_sim2, min=0)
    # cosine_sim1 = torch.where(cosine_sim1 > 0, cosine_sim1, 0)
    # cosine_sim2 = torch.where(cosine_sim2 > 0, cosine_sim2, 0)
    intersect_points_num1 = torch.sum(cosine_sim1 != 0, dim=0).int().item()
    intersect_points_num2 = torch.sum(cosine_sim2 != 0, dim=0).int().item()
    if intersect_points_num1 != 0 or intersect_points_num2 != 0:
        global intersect_num_total
        intersect_num_total += intersect_points_num1
        intersect_num_total += intersect_points_num2
        print("intersect_points_num1: {}".format(intersect_points_num1))
        print("intersect_points_num2: {}".format(intersect_points_num2))

    global data_len
    data_len += 2
    print("data_len: {}".format(data_len))

    if not specs.get("visualize"):
        return

    if intersect_points_num1 != 0:
        interact_points1 = (cosine_sim1 != 0).numpy()
        closest_center1 = closest_center1.numpy()
        min_indices1 = min_indices1.numpy()
        direction_pred1 = direction_pred1.numpy()
        arrow_pred_list = []
        arrow_prior_list = []
        for i in range(interact_points1.shape[0]):
            if interact_points1[i]:
                print("direction pred: {}".format(direction_pred1[i]))
                print("direction gt: {}".format(direction1[min_indices1[i]]))
                print("angle: {}".format(get_angle_degree(direction_pred1[i], direction1[min_indices1[i]])))
                pcd1.colors[i] = (0, 0, 1)
                arrow_pred = geometry_utils.get_arrow(direction_pred1[i], closest_center1[i],
                                                      np.linalg.norm(pcd1_np[i] - closest_center1[i]))
                arrow_pred.paint_uniform_color((0, 0, 1))
                arrow_pred_list.append(arrow_pred)

                arrow_prior = geometry_utils.get_arrow(direction1[min_indices1[i]], center1[min_indices1[i]], 0.5)
                arrow_prior.paint_uniform_color((1, 0, 0))
                arrow_prior_list.append(arrow_prior)

        o3d.visualization.draw_geometries([pcd1] + arrow_pred_list + arrow_prior_list, mesh_show_back_face=True)

    if intersect_points_num2 != 0:
        interact_points2 = (cosine_sim2 != 0).numpy()
        closest_center2 = closest_center2.numpy()
        min_indices2 = min_indices2.numpy()
        direction_pred2 = direction_pred2.numpy()
        arrow_pred_list = []
        arrow_prior_list = []
        for i in range(interact_points2.shape[0]):
            if interact_points2[i]:
                print("direction pred: {}".format(direction_pred2[i]))
                print("direction gt: {}".format(direction2[min_indices2[i]]))
                print("angle: {}".format(get_angle_degree(direction_pred2[i], direction2[min_indices2[i]])))
                pcd2.colors[i] = (0, 0, 1)
                arrow_pred = geometry_utils.get_arrow(direction_pred2[i], closest_center2[i],
                                                      np.linalg.norm(pcd2_np[i] - closest_center2[i]))
                arrow_pred.paint_uniform_color((0, 0, 1))
                arrow_pred_list.append(arrow_pred)

                arrow_prior = geometry_utils.get_arrow(direction2[min_indices2[i]], center2[min_indices2[i]], 0.5)
                arrow_prior.paint_uniform_color((1, 0, 0))
                arrow_prior_list.append(arrow_prior)

        o3d.visualization.draw_geometries([pcd2] + arrow_pred_list + arrow_prior_list, mesh_show_back_face=True)


def visualize(specs, filename):
    geometries_path = get_geometry_path(specs, filename)
    visualize_intersection(geometries_path)


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/ibsa_statistics.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree_dir = specs.get("path_options").get("filename_tree_dir")
    filename_tree = path_utils.get_filename_tree(specs,
                                                 specs.get("path_options").get("geometries_dir").get(filename_tree_dir))

    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                print('current file: ', filename)
                try:
                    visualize(specs, filename)
                except Exception as e:
                    print(e)

    print(intersect_num_total)
    print(data_len)
    print(intersect_num_total / (data_len * 2048))
