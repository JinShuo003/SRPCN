import json
import os
import re
from ordered_set import OrderedSet


def read_config(config_filepath: str):
    """
    读取json格式的配置文件
    Args:
        config_filepath: 配置文件路径
    Returns:
        dict格式的配置文件
    """
    if not os.path.isfile(config_filepath):
        raise Exception("The experiment config file does not exist")

    return json.load(open(config_filepath))


def get_filename_tree(specs: dict, base_path: str):
    """以base_path为基准构建文件树，文件树的格式为
    {
    'scene1': {
              'scene1.1000' :[scene1.1000_view0_0,
                              scene1.1000_view0_1],
              'scene1.1001' :[scene1.1001_view0_0,
                              scene1.1001_view0_1]
              }
    }
    """
    def sort_key(scene_name):
        numbers = re.findall(r'\d+', scene_name)
        return int(numbers[-1])

    # 构建文件树
    category_re = specs.get("path_options").get("format_info").get("category_re")
    scene_re = specs.get("path_options").get("format_info").get("scene_re")
    filename_re = specs.get("path_options").get("format_info").get("filename_re")

    handle_category = specs.get("path_options").get("format_info").get("handle_category")
    handle_scene = specs.get("path_options").get("format_info").get("handle_scene")
    handle_filename = specs.get("path_options").get("format_info").get("handle_filename")

    filename_tree = dict()
    folder_info = os.walk(base_path)
    for dir_path, dir_names, filenames in folder_info:
        # 顶级目录不做处理
        if dir_path == base_path:
            continue
        category = dir_path.split('\\')[-1]
        if not regular_match(handle_category, category):
            continue
        if category not in filename_tree:
            filename_tree[category] = dict()
        for filename in filenames:
            scene = re.match(scene_re, filename).group()
            if not regular_match(handle_scene, scene):
                continue
            if scene not in filename_tree[category]:
                filename_tree[category][scene] = OrderedSet()
            filename = re.match(filename_re, filename).group()
            if not regular_match(handle_filename, filename):
                continue
            filename_tree[category][scene].add(filename)
    # 将有序集合转为列表
    filename_tree_copy = dict()
    for category in filename_tree.keys():
        for scene in filename_tree[category]:
            filename_tree[category][scene] = list(filename_tree[category][scene])
            if len(filename_tree[category][scene]) != 0:
                if category not in filename_tree_copy.keys():
                    filename_tree_copy[category] = dict()
                filename_tree_copy[category][scene] = filename_tree[category][scene]
                filename_tree_copy[category][scene].sort(key=sort_key)
    return filename_tree_copy


def generate_path(path: str):
    if not isinstance(path, str):
        print("The type of path should be str")
    if not os.path.exists(path):
        os.makedirs(path)


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)


def get_geometries_path(specs: dict, scene):
    path_options = specs.get("path_options")

    category_re = path_options.get("format_info").get("category_re")
    scene_re = path_options.get("format_info").get("scene_re")
    category = re.match(category_re, scene).group()
    scene = re.match(scene_re, scene).group()

    is_mesh1_needed = False
    is_mesh2_needed = False
    is_pcd1_needed = False
    is_pcd2_needed = False
    is_ibs_mesh_needed = False
    is_ibs_pcd_needed = False
    if "mesh1" in path_options.get("required_geometry").keys() and path_options.get("required_geometry").get("mesh1"):
        is_mesh1_needed = True
    if "mesh2" in path_options.get("required_geometry").keys() and path_options.get("required_geometry").get("mesh2"):
        is_mesh2_needed = True
    if "pcd1" in path_options.get("required_geometry").keys() and path_options.get("required_geometry").get("pcd1"):
        is_pcd1_needed = True
    if "pcd2" in path_options.get("required_geometry").keys() and path_options.get("required_geometry").get("pcd2"):
        is_pcd2_needed = True
    if "ibs_mesh" in path_options.get("required_geometry").keys() and path_options.get("required_geometry").get("ibs_mesh"):
        is_ibs_mesh_needed = True
    if "ibs_pcd" in path_options.get("required_geometry").keys() and path_options.get("required_geometry").get("ibs_pcd"):
        is_ibs_pcd_needed = True

    geometries_path = dict()
    if is_mesh1_needed:
        mesh_dir = path_options.get("geometries_dir").get("mesh_dir")
        mesh1_filename = '{}_{}.obj'.format(scene, 0)
        geometries_path['mesh1'] = os.path.join(mesh_dir, category, mesh1_filename)
    if is_mesh2_needed:
        mesh_dir = path_options.get("geometries_dir").get("mesh_dir")
        mesh2_filename = '{}_{}.obj'.format(scene, 1)
        geometries_path['mesh2'] = os.path.join(mesh_dir, category, mesh2_filename)
    if is_pcd1_needed:
        pcd_dir = path_options.get("geometries_dir").get("pcd_dir")
        pcd1_filename = '{}_{}.ply'.format(scene, 0)
        geometries_path['pcd1'] = os.path.join(pcd_dir, category, pcd1_filename)
    if is_pcd2_needed:
        pcd_dir = path_options.get("geometries_dir").get("pcd_dir")
        pcd2_filename = '{}_{}.ply'.format(scene, 1)
        geometries_path['pcd2'] = os.path.join(pcd_dir, category, pcd2_filename)
    if is_ibs_mesh_needed:
        ibs_mesh_dir = path_options.get("geometries_dir").get("ibs_mesh_dir")
        ibs_mesh_filename = '{}.obj'.format(scene)
        geometries_path['ibs_mesh'] = os.path.join(ibs_mesh_dir, category, ibs_mesh_filename)
    if is_ibs_pcd_needed:
        ibs_pcd_dir = path_options.get("geometries_dir").get("ibs_pcd_dir")
        ibs_pcd_filename = '{}.ply'.format(scene)
        geometries_path['ibs_pcd'] = os.path.join(ibs_pcd_dir, category, ibs_pcd_filename)

    return geometries_path
