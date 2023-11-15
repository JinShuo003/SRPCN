import json
import os
import re
from ordered_set import OrderedSet


def parseConfig(config_filepath: str):
    with open(config_filepath, 'r') as configfile:
        config = json.load(configfile)
        return config


def getFilenameTree(specs: dict, base_path: str):
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
    # 构建文件树
    category_re = specs["category_re"]
    scene_re = specs["scene_re"]
    filename_re = specs["filename_re"]

    handle_category = specs["handle_category"]
    handle_scene = specs["handle_scene"]
    handle_filename = specs["handle_filename"]

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
    return filename_tree_copy


def generatePath(specs: dict, path_list: list):
    """检查specs中的path是否存在，不存在则创建"""
    for path in path_list:
        if not os.path.isdir(specs[path]):
            os.makedirs(specs[path])


def regular_match(regExp: str, target: str):
    return re.match(regExp, target)