"""
可视化工具
"""
import json
import os
import re

import numpy as np
import open3d as o3d

from utils import geometry_utils, path_utils


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
    pcd_gt_dir = specs.get("path_options").get("geometries_dir").get("pcd_gt_dir")
    pcd_scan_dir = specs.get("path_options").get("geometries_dir").get("pcd_scan_dir")
    pcd_pred_dir = specs.get("path_options").get("geometries_dir").get("pcd_pred_dir")
    medial_axis_sphere_dir = specs.get("path_options").get("geometries_dir").get("medial_axis_sphere_dir")

    mesh1_filename = '{}_{}.obj'.format(scene, 0)
    mesh2_filename = '{}_{}.obj'.format(scene, 1)
    ibs_mesh_gt_filename = '{}.obj'.format(scene)
    ibs_pcd_gt_filename = '{}.ply'.format(scene)
    ibs1_pcd_gt_filename = '{}_{}.ply'.format(scene, 0)
    ibs2_pcd_gt_filename = '{}_{}.ply'.format(scene, 1)
    pcd1_gt_filename = '{}_{}.ply'.format(scene, 0)
    pcd2_gt_filename = '{}_{}.ply'.format(scene, 1)
    pcd1_scan_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_scan_filename = '{}_{}.ply'.format(filename, 1)
    pcd1_pred_filename = '{}_{}.ply'.format(filename, 0)
    pcd2_pred_filename = '{}_{}.ply'.format(filename, 1)
    medial_axis_sphere_filename = '{}.npz'.format(scene)
    medial_axis_sphere1_filename = '{}_{}.npz'.format(scene, 0)
    medial_axis_sphere2_filename = '{}_{}.npz'.format(scene, 1)

    geometry_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    geometry_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    geometry_path['ibs_mesh_gt'] = os.path.join(ibs_mesh_gt_dir, category, ibs_mesh_gt_filename)
    geometry_path['ibs_pcd_gt'] = os.path.join(ibs_pcd_gt_dir, category, ibs_pcd_gt_filename)
    geometry_path['ibs1_pcd_gt'] = os.path.join(ibs_pcd_gt_dir, category, ibs1_pcd_gt_filename)
    geometry_path['ibs2_pcd_gt'] = os.path.join(ibs_pcd_gt_dir, category, ibs2_pcd_gt_filename)
    geometry_path['pcd1_gt'] = os.path.join(pcd_gt_dir, category, pcd1_gt_filename)
    geometry_path['pcd2_gt'] = os.path.join(pcd_gt_dir, category, pcd2_gt_filename)
    geometry_path['pcd1_scan'] = os.path.join(pcd_scan_dir, category, pcd1_scan_filename)
    geometry_path['pcd2_scan'] = os.path.join(pcd_scan_dir, category, pcd2_scan_filename)
    geometry_path['pcd1_pred'] = os.path.join(pcd_pred_dir, category, pcd1_pred_filename)
    geometry_path['pcd2_pred'] = os.path.join(pcd_pred_dir, category, pcd2_pred_filename)
    geometry_path['medial_axis_sphere'] = os.path.join(medial_axis_sphere_dir, category, medial_axis_sphere_filename)
    geometry_path['medial_axis_sphere1'] = os.path.join(medial_axis_sphere_dir, category, medial_axis_sphere1_filename)
    geometry_path['medial_axis_sphere2'] = os.path.join(medial_axis_sphere_dir, category, medial_axis_sphere2_filename)

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


class meshHandler(GeometryHandler):
    def __init__(self):
        super(meshHandler, self).__init__()
        pass

    def read_geometry(self, mesh_path):
        return o3d.io.read_triangle_mesh(mesh_path)

    def color_geometry(self, mesh, mesh_color):
        mesh.paint_uniform_color(mesh_color)

    def get(self, mesh_path, mesh_color, mesh_option):
        mesh = super().get(mesh_path, mesh_color, mesh_option)
        return mesh


class pcdHandler(GeometryHandler):
    def __init__(self):
        super(pcdHandler, self).__init__()
        pass

    def read_geometry(self, pcd_path):
        return o3d.io.read_point_cloud(pcd_path)

    def color_geometry(self, pcd, pcd_color):
        pcd.paint_uniform_color(pcd_color)

    def get(self, pcd_path, pcd_color, pcd_option):
        pcd = super().get(pcd_path, pcd_color, pcd_option)
        return pcd


class medialAxisSphereHandler(GeometryHandler):
    def __init__(self):
        super(medialAxisSphereHandler, self).__init__()
        pass

    def read_geometry(self, medial_axis_sphere_path):
        data = np.load(medial_axis_sphere_path)
        center, radius = data["center"], data["radius"]
        sphere_list = []
        for i in range(center.shape[0]):
            sphere = geometry_utils.get_sphere_mesh(center[i], radius[i])
            sphere_list.append(sphere)
            # if i == 100:
            #     break
        return sphere_list

    def color_geometry(self, medial_axis_sphere, sphere_color):
        for sphere in medial_axis_sphere:
            sphere.paint_uniform_color(sphere_color)

    def get(self, medial_axis_sphere_path, sphere_color, sphere_option):
        sphere_list = super().get(medial_axis_sphere_path, sphere_color, sphere_option)
        return sphere_list


class medialAxisDirectionHandler(GeometryHandler):
    def __init__(self, mode: str, idx: int):
        super(medialAxisDirectionHandler, self).__init__()
        self.mode = mode
        self.idx = idx

    def read_geometry(self, medial_axis_sphere_path):
        data = np.load(medial_axis_sphere_path)
        if self.mode == "origin":
            center, radius, direction1, direction2 = data["center"], data["radius"], data["direction1"], data[
                "direction2"]
            if self.idx == 1:
                direction = direction1
            elif self.idx == 2:
                direction = direction2
        elif self.mode == "norm":
            center, radius, direction = data["center"], data["radius"], data["direction"]
        arrow_list = []
        for i in range(center.shape[0]):
            arrow = geometry_utils.get_arrow(direction[i], center[i], 0.05)
            arrow_list.append(arrow)
            # if i == 100:
            #     break

        # 组合所有箭头
        vertices = None
        faces = None
        vertices_num = 0
        for arrow in arrow_list:
            vertices_cur = np.array(arrow.vertices)
            faces_cur = np.array(arrow.triangles)
            if vertices is None:
                vertices = vertices_cur
            else:
                vertices = np.vstack((vertices, vertices_cur))
            if faces is None:
                faces = faces_cur
            else:
                faces = np.vstack((faces, faces_cur+vertices_num))
            vertices_num = vertices.shape[0]

        arrow = o3d.geometry.TriangleMesh()
        arrow.vertices = o3d.utility.Vector3dVector(vertices)
        arrow.triangles = o3d.utility.Vector3iVector(faces)
        # o3d.visualization.draw_geometries([arrow])

        o3d.io.write_triangle_mesh("arrow{}.obj".format(self.idx), arrow)
        return arrow_list

    def color_geometry(self, medial_axis_sphere, sphere_color):
        for sphere in medial_axis_sphere:
            sphere.paint_uniform_color(sphere_color)

    def get(self, medial_axis_sphere_path, color, option):
        arrow_list = super().get(medial_axis_sphere_path, color, option)
        return arrow_list


def visualize(specs, filename):
    batch_geometry_key_set = set(["medial_axis_sphere",
                                  "medial_axis_sphere1",
                                  "medial_axis_sphere2",
                                  "medial_axis_origin_direction1",
                                  "medial_axis_origin_direction2",
                                  "medial_axis_direction1",
                                  "medial_axis_direction2"])

    container = dict()
    geometries = []
    geometry_path = getGeometryPath(specs, filename)
    geometry_color = getGeometryColor(specs)
    geometry_option = getGeometryOption(specs)

    mesh1 = meshHandler().get(geometry_path["mesh1"], geometry_color["mesh1"], geometry_option["mesh1"])
    mesh2 = meshHandler().get(geometry_path["mesh2"], geometry_color["mesh2"], geometry_option["mesh2"])
    ibs_mesh_gt = meshHandler().get(geometry_path["ibs_mesh_gt"], geometry_color["ibs_mesh_gt"], geometry_option["ibs_mesh_gt"])
    ibs_pcd_gt = pcdHandler().get(geometry_path["ibs_pcd_gt"], geometry_color["ibs_pcd_gt"], geometry_option["ibs_pcd_gt"])
    ibs1_pcd_gt = pcdHandler().get(geometry_path["ibs1_pcd_gt"], geometry_color["ibs1_pcd_gt"], geometry_option["ibs1_pcd_gt"])
    ibs2_pcd_gt = pcdHandler().get(geometry_path["ibs2_pcd_gt"], geometry_color["ibs2_pcd_gt"], geometry_option["ibs2_pcd_gt"])
    pcd1_gt = pcdHandler().get(geometry_path['pcd1_gt'], geometry_color['pcd1_gt'], geometry_option["pcd1_gt"])
    pcd2_gt = pcdHandler().get(geometry_path['pcd2_gt'], geometry_color['pcd2_gt'], geometry_option["pcd2_gt"])
    pcd1_scan = pcdHandler().get(geometry_path['pcd1_scan'], geometry_color['pcd1_scan'], geometry_option["pcd1_scan"])
    pcd2_scan = pcdHandler().get(geometry_path['pcd2_scan'], geometry_color['pcd2_scan'], geometry_option["pcd2_scan"])
    pcd1_pred = pcdHandler().get(geometry_path['pcd1_pred'], geometry_color['pcd1_pred'], geometry_option["pcd1_pred"])
    pcd2_pred = pcdHandler().get(geometry_path['pcd2_pred'], geometry_color['pcd2_pred'], geometry_option["pcd2_pred"])
    medial_axis_sphere = medialAxisSphereHandler().get(geometry_path['medial_axis_sphere'], geometry_color['medial_axis_sphere'], geometry_option["medial_axis_sphere"])
    medial_axis_sphere1 = medialAxisSphereHandler().get(geometry_path['medial_axis_sphere1'], geometry_color['medial_axis_sphere1'], geometry_option["medial_axis_sphere1"])
    medial_axis_sphere2 = medialAxisSphereHandler().get(geometry_path['medial_axis_sphere2'], geometry_color['medial_axis_sphere2'], geometry_option["medial_axis_sphere2"])
    medial_axis_origin_direction1 = medialAxisDirectionHandler("origin", 1).get(geometry_path['medial_axis_sphere'], geometry_color['medial_axis_direction1'], geometry_option["medial_axis_origin_direction1"])
    medial_axis_origin_direction2 = medialAxisDirectionHandler("origin", 2).get(geometry_path['medial_axis_sphere'], geometry_color['medial_axis_direction2'], geometry_option["medial_axis_origin_direction2"])
    medial_axis_direction1 = medialAxisDirectionHandler("norm", 1).get(geometry_path['medial_axis_sphere1'], geometry_color['medial_axis_direction1'], geometry_option["medial_axis_direction1"])
    medial_axis_direction2 = medialAxisDirectionHandler("norm", 2).get(geometry_path['medial_axis_sphere2'], geometry_color['medial_axis_direction2'], geometry_option["medial_axis_direction2"])

    coord_frame = geometry_utils.get_coordinate(size=0.5)
    unit_sphere_pcd = geometry_utils.get_sphere_pcd(radius=0.5)

    container['mesh1'] = mesh1
    container['mesh2'] = mesh2
    container['ibs_mesh_gt'] = ibs_mesh_gt
    container['ibs_pcd_gt'] = ibs_pcd_gt
    container['ibs1_pcd_gt'] = ibs1_pcd_gt
    container['ibs2_pcd_gt'] = ibs2_pcd_gt
    container['pcd1_gt'] = pcd1_gt
    container['pcd2_gt'] = pcd2_gt
    container['pcd1_scan'] = pcd1_scan
    container['pcd2_scan'] = pcd2_scan
    container['pcd1_pred'] = pcd1_pred
    container['pcd2_pred'] = pcd2_pred
    container['medial_axis_sphere'] = medial_axis_sphere
    container['medial_axis_sphere1'] = medial_axis_sphere1
    container['medial_axis_sphere2'] = medial_axis_sphere2
    container['medial_axis_origin_direction1'] = medial_axis_origin_direction1
    container['medial_axis_origin_direction2'] = medial_axis_origin_direction2
    container['medial_axis_direction1'] = medial_axis_direction1
    container['medial_axis_direction2'] = medial_axis_direction2
    container['coord_frame'] = coord_frame
    container['unit_sphere'] = unit_sphere_pcd

    for key in geometry_option.keys():
        if geometry_option[key] is True:
            if key in batch_geometry_key_set:
                geometries += container[key]
            else:
                geometries.append(container[key])

    o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True, mesh_show_back_face=True)


if __name__ == '__main__':
    # 获取配置参数
    config_filepath = 'configs/geometry_visualizer.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree_dir = specs.get("path_options").get("filename_tree_dir")
    filename_tree = path_utils.get_filename_tree(specs,
                                                 specs.get("path_options").get("geometries_dir").get(filename_tree_dir))

    for category in filename_tree:
        for scene in filename_tree[category]:
            print('current scene1: ', scene)
            for filename in filename_tree[category][scene]:
                print('current file: ', filename)
                try:
                    visualize(specs, filename)
                except Exception as e:
                    print(e)
