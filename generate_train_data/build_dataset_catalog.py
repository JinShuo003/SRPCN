"""
划分测试集、训练集，构造数据集的目录文件
"""
import json
import os
import re

import torch
from torch.utils.data import DataLoader

from utils import path_utils


def serialize_data(specs):
    pcd_scan_path = specs["pcd_scan_path"]
    scenename_re = specs["scenename_re"]
    filename_re = specs["filename_re"]
    categories = specs["categories"]
    data = {}
    scenename_list = []
    for category in categories:
        data[category] = {}
        filename_list = os.listdir(os.path.join(pcd_scan_path, category))
        handled_data = set()
        for filename in filename_list:
            scenename = re.match(scenename_re, filename).group()
            if not scenename in data[category].keys():
                data[category][scenename] = []
            data_name = re.match(filename_re, filename).group()
            if data_name in handled_data:
                continue
            data[category][scenename].append(data_name)
            handled_data.add(data_name)

    for category in data.keys():
        scenename_cur_list = []
        for scene in data[category].keys():
            scenename_cur_list.append(scene)
        scenename_list.append(scenename_cur_list)
    return data, scenename_list


def partition_dataset(data, scenename_list, specs):
    """按比例划分每个类别下的场景，然后将对应场景的所有数据划分到训练集或测试集"""
    category_re = specs["category_re"]
    train_scene_split = []
    test_scene_split = []

    # 获取场景划分结果
    for i in range(len(scenename_list)):
        # 计算训练集、测试集、验证集大小
        train_size = int(len(scenename_list[i]) * specs["partition_option"]["train_dataset_proportion"])
        test_size = len(scenename_list[i]) - train_size
        print("train: {}\ntest: {}\n".format(train_size, test_size))

        train_scene, test_scene = torch.utils.data.random_split(scenename_list[i], [train_size, test_size])
        train_scene = [train_scene.dataset[i] for i in train_scene.indices]
        test_scene = [test_scene.dataset[i] for i in test_scene.indices]
        train_scene_split.append(train_scene)
        test_scene_split.append(test_scene)

    # 根据场景划分结果将对应数据全部放入训练集或测试集
    train_dataset = {}
    test_dataset = {}
    for i in range(len(train_scene_split)):
        category = re.match(category_re, train_scene_split[i][0]).group()
        train_dataset[category] = []
        for scene in train_scene_split[i]:
            train_dataset[category] += data[category][scene]
    for i in range(len(test_scene_split)):
        category = re.match(category_re, test_scene_split[i][0]).group()
        test_dataset[category] = []
        for scene in test_scene_split[i]:
            test_dataset[category] += data[category][scene]
    return train_dataset, test_dataset


def generate_split_file(dataset_path, dir_name, filename, dataset, specs):
    """根据数据集划分结果生成对应的文件存储划分信息"""
    dataset_name = specs["dataset_name"]
    # 创建目录
    if not os.path.isdir(os.path.join(dataset_path, dir_name)):
        os.mkdir(os.path.join(dataset_path, dir_name))

    # 生成数据
    split_data = {dataset_name: dataset}
    split_json = json.dumps(split_data, indent=1)

    # 生成文件
    split_path = os.path.join(dataset_path, dir_name, "{}.json".format(filename))
    if os.path.isfile(split_path):
        os.remove(split_path)
    with open(split_path, 'w', newline='\n') as f:
        f.write(split_json)

    # 写入每个场景的split文件
    for key in split_data[dataset_name]:
        scene_data = {dataset_name: {}}
        scene_data[dataset_name][key] = dataset[key]
        scene_json = json.dumps(scene_data, indent=1)
        # 生成文件
        scene_path = os.path.join(dataset_path, dir_name, "{}_{}.json".format(filename, key))
        if os.path.isfile(scene_path):
            os.remove(scene_path)
        with open(scene_path, 'w', newline='\n') as f:
            f.write(scene_json)


if __name__ == '__main__':
    # 获取配置参数
    configFile_path = 'configs/build_dataset_catalog.json'
    specs = path_utils.read_config(configFile_path)

    dataset_catalog_save_path = specs["dataset_catalog_save_path"]
    train_split_dirname = specs["train_split_dirname"]
    test_split_dirname = specs["test_split_dirname"]
    categories = specs["categories"]

    # 若目录不存在则创建目录
    if not os.path.isdir(specs["dataset_catalog_save_path"]):
        os.mkdir(specs["dataset_catalog_save_path"])

    # 将所有数据序列化为字典，格式为{"scene*": {scene*.****: {scene*.****_view*}}}
    data, scenename_list = serialize_data(specs)
    # 按场景名划分训练集、测试集
    train_dataset, test_dataset = partition_dataset(data, scenename_list, specs)
    # 将划分结果写成文件
    generate_split_file(dataset_catalog_save_path, train_split_dirname, "train", train_dataset, specs)
    generate_split_file(dataset_catalog_save_path, test_split_dirname, "test", test_dataset, specs)
