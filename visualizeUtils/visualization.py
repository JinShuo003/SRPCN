"""
可视化工具，配置好./config/visualization.json后可以可视化mesh模型、点云、位于模型表面和IBS表面的sdf点、各自和总体的aabb框、交互区域gt
"""
import json
import os
import re

import open3d as o3d

from utils import geometry_utils, path_utils


def parseConfig(config_filepath: str = './visualization.json'):
    with open(config_filepath, 'r') as configfile:
        specs = json.load(configfile)

    return specs


def getGeometryPath(specs, filename):
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    category_re = specs.get("path_options").get("format_info").get("category_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")
    category = re.match(category_re, filename).group()
    scene = re.match(scene_re, filename).group()
    filename = re.match(filename_re, filename).group()

    geometry_path = dict()

    mesh_dir = specs.get("path_options").get("geometries_dir").get("mesh_dir")
    ibs_mesh_gt_dir = specs.get("path_options").get("geometries_dir").get("ibs_mesh_gt_dir")
    ibs_pcd_gt_dir = specs.get("path_options").get("geometries_dir").get("ibs_pcd_gt_dir")
    ibs_pcd_pred_dir = specs.get("path_options").get("geometries_dir").get("ibs_pcd_pred_dir")
    pcd_gt_dir = specs.get("path_options").get("geometries_dir").get("pcd_gt_dir")
    pcd_scan_dir = specs.get("path_options").get("geometries_dir").get("pcd_scan_dir")
    pcd_pred_dir = specs.get("path_options").get("geometries_dir").get("pcd_pred_dir")

    mesh1_filename = '{}_{}.obj'.format(scene, 0)
    mesh2_filename = '{}_{}.obj'.format(scene, 1)
    ibs_mesh_gt_filename = '{}.obj'.format(scene)
    ibs_pcd_gt_filename = '{}.ply'.format(scene)
    ibs_pcd_pred_filename = '{}.ply'.format(filename)
    pcd1_gt_filename = '{}_{}.ply'.format(scene, 0)
    pcd2_gt_filename = '{}_{}.ply'.format(scene, 1)
    pcd1_scan_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_scan_filename = '{}_{}.ply'.format(filename, 1)
    pcd1_pred_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_pred_filename = '{}_{}.ply'.format(filename, 1)

    geometry_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometry_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometry_path['ibs_mesh_gt'] = os.path.join(ibs_mesh_gt_dir, category, ibs_mesh_gt_filename)
    geometry_path['ibs_pcd_gt'] = os.path.join(ibs_pcd_gt_dir, category, ibs_pcd_gt_filename)
    geometry_path['ibs_pcd_pred'] = os.path.join(ibs_pcd_pred_dir, category, ibs_pcd_pred_filename)
    geometry_path['pcd1_gt'] = os.path.join(pcd_gt_dir, category, pcd1_gt_filename)
    geometry_path['pcd2_gt'] = os.path.join(pcd_gt_dir, category, pcd2_gt_filename)
    geometry_path['pcd1_scan'] = os.path.join(pcd_scan_dir, category, pcd1_scan_filename)
    geometry_path['pcd2_scan'] = os.path.join(pcd_scan_dir, category, pcd2_scan_filename)
    geometry_path['pcd1_pred'] = os.path.join(pcd_pred_dir, category, pcd1_pred_filename)
    geometry_path['pcd2_pred'] = os.path.join(pcd_pred_dir, category, pcd2_pred_filename)
    
    return geometry_path


def getGeometryColor(specs):
    geometry_color_dict = specs["visualization_options"]["colors"]
    for key in geometry_color_dict.keys():
        geometry_color_dict[key] = tuple(geometry_color_dict[key])
    return geometry_color_dict


def getGeometryOption(specs):
    geometry_color_dict = specs["visualization_options"]["geometries"]
    return geometry_color_dict


class GeometryHandler:
    def __init__(self):
        pass

    def read_geometry(self, geometry_path):
        pass

    def color_geometry(self, geometry, color):
        pass

    def get(self, path, color, option):
        if option is False:
            return None
        geometry = self.read_geometry(path)
        self.color_geometry(geometry, color)
        return geometry


class meshGetter(GeometryHandler):
    def __init__(self):
        super(meshGetter, self).__init__()
        pass

    def read_geometry(self, mesh_path):
        return o3d.io.read_triangle_mesh(mesh_path)

    def color_geometry(self, mesh, mesh_color):
        mesh.paint_uniform_color(mesh_color)

    def get(self, mesh_path, mesh_color, mesh_option):
        mesh = super().get(mesh_path, mesh_color, mesh_option)
        return mesh


class pcdGetter(GeometryHandler):
    def __init__(self):
        super(pcdGetter, self).__init__()
        pass

    def read_geometry(self, pcd_path):
        return o3d.io.read_point_cloud(pcd_path)

    def color_geometry(self, pcd, pcd_color):
        pcd.paint_uniform_color(pcd_color)

    def get(self, pcd_path, pcd_color, pcd_option):
        pcd = super().get(pcd_path, pcd_color, pcd_option)
        return pcd


def visualize(specs, filename):
    container = dict()
    geometries = []
    geometry_path = getGeometryPath(specs, filename)
    geometry_color = getGeometryColor(specs)
    geometry_option = getGeometryOption(specs)

    mesh1 = meshGetter().get(geometry_path["mesh1"], geometry_color["mesh1"], geometry_option["mesh1"])
    mesh2 = meshGetter().get(geometry_path["mesh2"], geometry_color["mesh2"], geometry_option["mesh2"])
    ibs_mesh_gt = meshGetter().get(geometry_path["ibs_mesh_gt"], geometry_color["ibs_mesh_gt"], geometry_option["ibs_mesh_gt"])
    ibs_pcd_gt = pcdGetter().get(geometry_path["ibs_pcd_gt"], geometry_color["ibs_pcd_gt"], geometry_option["ibs_pcd_gt"])
    ibs_pcd_pred = pcdGetter().get(geometry_path["ibs_pcd_pred"], geometry_color["ibs_pcd_pred"], geometry_option["ibs_pcd_pred"])
    pcd1_gt = pcdGetter().get(geometry_path['pcd1_gt'], geometry_color['pcd1_gt'], geometry_option["pcd1_gt"])
    pcd2_gt = pcdGetter().get(geometry_path['pcd2_gt'], geometry_color['pcd2_gt'], geometry_option["pcd2_gt"])
    pcd1_scan = pcdGetter().get(geometry_path['pcd1_scan'], geometry_color['pcd1_scan'], geometry_option["pcd1_scan"])
    pcd2_scan = pcdGetter().get(geometry_path['pcd2_scan'], geometry_color['pcd2_scan'], geometry_option["pcd2_scan"])
    pcd1_pred = pcdGetter().get(geometry_path['pcd1_pred'], geometry_color['pcd1_pred'], geometry_option["pcd1_pred"])
    pcd2_pred = pcdGetter().get(geometry_path['pcd2_pred'], geometry_color['pcd2_pred'], geometry_option["pcd2_pred"])

    coord_frame = geometry_utils.get_unit_coordinate(size=1)
    unit_sphere_pcd = geometry_utils.get_sphere_pcd(radius=1)

    container['mesh1'] = mesh1
    container['mesh2'] = mesh2
    container['ibs_mesh_gt'] = ibs_mesh_gt
    container['ibs_pcd_gt'] = ibs_pcd_gt
    container['ibs_pcd_pred'] = ibs_pcd_pred
    container['pcd1_gt'] = pcd1_gt
    container['pcd2_gt'] = pcd2_gt
    container['pcd1_scan'] = pcd1_scan
    container['pcd2_scan'] = pcd2_scan
    container['pcd1_pred'] = pcd1_pred
    container['pcd2_pred'] = pcd2_pred
    container['coord_frame'] = coord_frame
    container['unit_sphere'] = unit_sphere_pcd

    for key in geometry_option.keys():
        if geometry_option[key] is True:
            geometries.append(container[key])

    o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True, mesh_show_back_face=True)


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'visualization.json'
    specs = parseConfig(config_filepath)

    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))

    for category in filename_tree:
        for scene in filename_tree[category]:
            print('current scene: ', scene)
            for filename in filename_tree[category][scene]:
                print('current file: ', filename)
                try:
                    visualize(specs, filename)
                except Exception:
                    pass
