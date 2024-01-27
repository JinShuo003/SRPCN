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


def handle_data(geometries_path: dict, tag: str):
    pcd = geometry_utils.read_point_cloud(geometries_path["pcd{}".format(tag)])
    pcd.paint_uniform_color((0.7, 0.3, 0.3))
    pcd_np = np.asarray(pcd.points)
    pcd_tensor = torch.from_numpy(pcd_np)

    center, radius, direction = geometry_utils.read_medial_axis_sphere_single(geometries_path["Medial_axis_sphere{}".format(tag)])
    center_tensor, radius_tensor, direction_tensor = torch.from_numpy(center), torch.from_numpy(radius), torch.from_numpy(direction)

    distances = torch.cdist(pcd_tensor, center_tensor, p=2)
    min_indices = torch.argmin(distances, dim=1)
    closest_center = center_tensor[min_indices, :]
    closest_direction = direction_tensor[min_indices, :]
    closest_radius = radius_tensor[min_indices]
    min_distances = distances[torch.arange(distances.shape[0]), min_indices]
    direction_pred = pcd_tensor - closest_center
    direction_pred /= torch.norm(direction_pred, dim=1, keepdim=True)
    cosine_sim = -F.cosine_similarity(direction_pred, closest_direction, dim=1) + np.cos(np.deg2rad(120))
    cosine_sim = torch.where(min_distances < 1.5 * closest_radius, cosine_sim, 0)
    cosine_sim = torch.clamp(cosine_sim, min=0)
    intersect_points_num = torch.sum(cosine_sim != 0, dim=0).int().item()
    if intersect_points_num != 0:
        global intersect_num_total
        intersect_num_total += intersect_points_num
        print("intersect_points_num: {}".format(intersect_points_num))

    global data_len
    data_len += 1

    if specs.get("visualize"):
        interact_points = (cosine_sim != 0).numpy()
        closest_center = closest_center.numpy()
        closest_direction = closest_direction.numpy()
        direction_pred = direction_pred.numpy()
        arrow_pred_list = []
        arrow_prior_list = []
        for i in range(interact_points.shape[0]):
            if interact_points[i]:
                print("direction pred: {}".format(direction_pred[i]))
                print("direction gt: {}".format(closest_direction[i]))
                print("angle: {}".format(get_angle_degree(direction_pred[i], closest_direction[i])))
                pcd.colors[i] = (0, 0, 1)
                arrow_pred = geometry_utils.get_arrow(direction_pred[i], closest_center[i],
                                                      np.linalg.norm(pcd_np[i] - closest_center[i]))
                arrow_pred.paint_uniform_color((0, 0, 1))
                arrow_pred_list.append(arrow_pred)

                arrow_prior = geometry_utils.get_arrow(closest_direction[i], closest_center[i], 0.5)
                arrow_prior.paint_uniform_color((1, 0, 0))
                arrow_prior_list.append(arrow_prior)

        o3d.visualization.draw_geometries([pcd] + arrow_pred_list + arrow_prior_list, mesh_show_back_face=True)


def visualize(specs, filename):
    geometries_path = get_geometry_path(specs, filename)
    handle_data(geometries_path, "1")
    handle_data(geometries_path, "2")


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/ibsa_visualizer.json'
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
