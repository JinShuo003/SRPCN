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

    pcd_pred_dir = specs.get("path_options").get("geometries_dir").get("pcd_pred_dir")
    IBS_mesh_dir = specs.get("path_options").get("geometries_dir").get("IBS_mesh_dir")
    IBS_pcd_dir = specs.get("path_options").get("geometries_dir").get("IBS_pcd_dir")
    Medial_axis_sphere_dir = specs.get("path_options").get("geometries_dir").get("Medial_axis_sphere_dir")
    Normalize_data_dir = specs.get("path_options").get("geometries_dir").get("Normalize_data_dir")

    pcd1_pred_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_pred_filename = '{}_{}.ply'.format(filename, 1)
    IBS_mesh_filename = '{}.obj'.format(scene)
    IBS_pcd1_filename = '{}_{}.ply'.format(scene, 0)
    IBS_pcd2_filename = '{}_{}.ply'.format(scene, 1)
    Medial_axis_sphere1_filename = '{}_0.npz'.format(scene, 0)
    Medial_axis_sphere2_filename = '{}_1.npz'.format(scene, 1)
    normalize_data1_filename = '{}_0.txt'.format(scene, 0)
    normalize_data2_filename = '{}_1.txt'.format(scene, 1)

    geometries_path['pcd1_pred'] = os.path.join(pcd_pred_dir, category, pcd1_pred_filename)
    geometries_path['pcd2_pred'] = os.path.join(pcd_pred_dir, category, pcd2_pred_filename)
    geometries_path['IBS_mesh'] = os.path.join(IBS_mesh_dir, category, IBS_mesh_filename)
    geometries_path['IBS_pcd1'] = os.path.join(IBS_pcd_dir, category, IBS_pcd1_filename)
    geometries_path['IBS_pcd2'] = os.path.join(IBS_pcd_dir, category, IBS_pcd2_filename)
    geometries_path['Medial_axis_sphere1'] = os.path.join(Medial_axis_sphere_dir, category, Medial_axis_sphere1_filename)
    geometries_path['Medial_axis_sphere2'] = os.path.join(Medial_axis_sphere_dir, category, Medial_axis_sphere2_filename)
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


def visualize_intersection(geometries_path: dict, tag: str):
    pcd_pred = geometry_utils.read_point_cloud(geometries_path["pcd{}_pred".format(tag)])
    pcd_pred.paint_uniform_color((0.7, 0.3, 0.3))
    translate, scale = get_normalize_para(geometries_path["normalize_data{}".format(tag)])
    pcd_pred = geometry_utils.geometry_transform(pcd_pred, np.array(translate), scale)

    IBS_mesh = geometry_utils.read_mesh(geometries_path["IBS_mesh"])
    IBS_mesh.paint_uniform_color((0.3, 0.7, 0.3))
    IBS_mesh = geometry_utils.geometry_transform(IBS_mesh, np.array(translate), scale)

    center, radius, direction = geometry_utils.read_medial_axis_sphere_single(geometries_path["Medial_axis_sphere{}".format(tag)])
    center_tensor, radius_tensor, direction_tensor = torch.from_numpy(center).unsqueeze(0), torch.from_numpy(radius).unsqueeze(0), torch.from_numpy(direction).unsqueeze(0)

    pcd_pred_tensor = torch.from_numpy(np.asarray(pcd_pred.points)).unsqueeze(0)
    distances = torch.cdist(pcd_pred_tensor, center_tensor, p=2)
    min_indices = torch.argmin(distances, dim=2)
    closest_center = torch.gather(center_tensor, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))
    closest_center_direction = torch.gather(direction_tensor, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))

    direction_pred = pcd_pred_tensor - closest_center
    cosine_sim = F.cosine_similarity(direction_pred, closest_center_direction, dim=2)
    loss = torch.clamp(-cosine_sim - 0.2, min=0)
    interact_points_num = torch.sum(loss != 0, dim=1).squeeze(0).float()  # (B)
    if interact_points_num == 0:
        print("no intersection")
        # return

    interact_points = (loss != 0).squeeze(0).numpy()
    closest_center = closest_center.squeeze(0).numpy()
    min_indices = min_indices.squeeze(0).numpy()
    direction_pred = direction_pred.squeeze(0).numpy()

    arrow_pred_list = []
    arrow_prior_list = []
    for i in range(interact_points.shape[0]):
        if interact_points[i]:
            pcd_pred.colors[i] = (0, 0, 1)
            arrow_pred = geometry_utils.get_arrow(direction_pred[i], closest_center[i], np.linalg.norm(np.asarray(pcd_pred.points)[i] - closest_center[i]))
            arrow_pred.paint_uniform_color((0, 0, 1))
            arrow_pred_list.append(arrow_pred)

            arrow_prior = geometry_utils.get_arrow(direction[min_indices[i]], center[min_indices[i]], 0.5)
            arrow_prior.paint_uniform_color((1, 0, 0))
            arrow_prior_list.append(arrow_prior)

    o3d.visualization.draw_geometries([pcd_pred, IBS_mesh] + arrow_pred_list + arrow_prior_list, mesh_show_back_face=True)


def visualize(specs, filename):
    visualize_obj1 = specs.get("visualize_obj1")
    visualize_obj2 = specs.get("visualize_obj2")
    geometries_path = get_geometry_path(specs, filename)

    assert visualize_obj1 or visualize_obj2

    if visualize_obj1:
        visualize_intersection(geometries_path, "1")
    if visualize_obj2:
        visualize_intersection(geometries_path, "2")


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
