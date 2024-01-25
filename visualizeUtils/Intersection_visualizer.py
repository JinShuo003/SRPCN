import torch

from utils import path_utils, geometry_utils
import open3d as o3d
import os
import re
import numpy as np
import torch.nn.functional as F


def get_geometry_path(specs, data_name):
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, data_name).group()
    scene = re.match(scene_re, data_name).group()
    filename = re.match(filename_re, data_name).group()

    geometries_path = dict()

    pcd_pred_dir = specs.get("path_options").get("geometries_dir").get("pcd_pred_dir")
    IBS_pcd_dir = specs.get("path_options").get("geometries_dir").get("IBS_pcd_dir")
    Medial_axis_sphere_dir = specs.get("path_options").get("geometries_dir").get("Medial_axis_sphere_dir")

    pcd1_pred_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_pred_filename = '{}_{}.ply'.format(filename, 1)
    IBS_pcd1_filename = '{}_{}.ply'.format(scene, 0)
    IBS_pcd2_filename = '{}_{}.ply'.format(scene, 1)
    Medial_axis_sphere1_filename = '{}_0.npz'.format(scene, 0)
    Medial_axis_sphere2_filename = '{}_1.npz'.format(scene, 1)

    geometries_path['pcd1_pred'] = os.path.join(pcd_pred_dir, category, pcd1_pred_filename)
    geometries_path['pcd2_pred'] = os.path.join(pcd_pred_dir, category, pcd2_pred_filename)
    geometries_path['IBS_pcd1'] = os.path.join(IBS_pcd_dir, category, IBS_pcd1_filename)
    geometries_path['IBS_pcd2'] = os.path.join(IBS_pcd_dir, category, IBS_pcd2_filename)
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


def visualize(specs, filename):
    geometries_path = get_geometry_path(specs, filename)

    sphere = geometry_utils.get_sphere_pcd(radius=0.5)
    pcd1_pred = geometry_utils.read_point_cloud(geometries_path["pcd1_pred"])
    pcd1_pred.paint_uniform_color((0.7, 0.3, 0.3))
    IBS_pcd1 = geometry_utils.read_point_cloud(geometries_path["IBS_pcd1"])
    IBS_pcd1.paint_uniform_color((0.3, 0.7, 0.3))
    center1, radius1, direction1 = geometry_utils.read_medial_axis_sphere_single(geometries_path["Medial_axis_sphere1"])
    center1_tensor, radius1_tensor, direction1_tensor = torch.from_numpy(center1).unsqueeze(0), torch.from_numpy(radius1).unsqueeze(0), torch.from_numpy(direction1).unsqueeze(0)

    pcd1_pred_tensor = torch.from_numpy(np.asarray(pcd1_pred.points)).unsqueeze(0)
    distances = torch.cdist(pcd1_pred_tensor, center1_tensor, p=2)
    min_indices = torch.argmin(distances, dim=2)
    closest_center = torch.gather(center1_tensor, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))
    closest_center_direction = torch.gather(direction1_tensor, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))

    direction_pred = pcd1_pred_tensor - closest_center
    cosine_sim = F.cosine_similarity(direction_pred, closest_center_direction, dim=2)
    loss = torch.clamp(-cosine_sim, min=0)

    interact_points = (loss != 0).squeeze(0).numpy()
    closest_center = closest_center.squeeze(0).numpy()
    min_indices = min_indices.squeeze(0).numpy()
    direction_pred = direction_pred.squeeze(0).numpy()

    for i in range(interact_points.shape[0]):
        if interact_points[i]:
            direction_pred[i] /= np.linalg.norm(direction_pred[i])
            print("i: {}".format(i))
            print("pred: {}".format(direction_pred[i]))
            print("gt: {}".format(direction1[min_indices[i]]))
            print("angle degree: {}".format(get_angle_degree(direction_pred[i], direction1[min_indices[i]])))
            print("cosine_sim: {}".format(F.cosine_similarity(torch.from_numpy(direction_pred[i]), torch.from_numpy(direction1[min_indices[i]]), dim=0)))

            pcd1_pred.colors[i] = (0, 0, 1)
            arrow_pred = geometry_utils.get_arrow(direction_pred[i], closest_center[i], 0.5)
            arrow_pred.paint_uniform_color((0, 0, 1))

            arrow_prior = geometry_utils.get_arrow(direction1[min_indices[i]], center1[min_indices[i]], 0.5)
            arrow_prior.paint_uniform_color((1, 0, 0))
            o3d.visualization.draw_geometries([pcd1_pred, IBS_pcd1, arrow_pred, arrow_prior])


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/Intersection_visualizer.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree_dir = specs.get("path_options").get("filename_tree_dir")
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get(filename_tree_dir))

    for category in filename_tree:
        for scene in filename_tree[category]:
            print('current scene1: ', scene)
            for filename in filename_tree[category][scene]:
                print('current file: ', filename)
                try:
                    visualize(specs, filename)
                except Exception as e:
                    print(e)