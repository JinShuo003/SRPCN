import logging
import os.path

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import json
import open3d as o3d
import numpy as np
import re

import networks.loss
from networks.models import *

import utils.data
import utils.workspace as ws
from utils.geometry_utils import get_pcd_from_np


def visualize_data(pcd1_path1, pcd2_path1, pcd1_path2, pcd2_path2, specs):
    # 将udf数据拆分开，并且转移到cpu
    pcd1_path1 = pcd1_path1.cpu().detach().numpy()
    pcd2_path1 = pcd2_path1.cpu().detach().numpy()
    pcd1_path2 = pcd1_path2.cpu().detach().numpy()
    pcd2_path2 = pcd2_path2.cpu().detach().numpy()

    for i in range(pcd1_path1.shape[0]):
        pcd1_1 = get_pcd_from_np(pcd1_path1[i])
        pcd2_1 = get_pcd_from_np(pcd2_path1[i])
        pcd1_2 = get_pcd_from_np(pcd1_path2[i])
        pcd2_2 = get_pcd_from_np(pcd2_path2[i])

        pcd1_1.paint_uniform_color([1, 0, 0])
        pcd2_1.paint_uniform_color([0, 1, 0])
        pcd1_2.paint_uniform_color([0, 0, 1])
        pcd2_2.paint_uniform_color([1, 0, 1])

        o3d.visualization.draw_geometries([pcd1_2, pcd2_2])


def get_dataloader(specs):
    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]
    scene_per_batch = specs["ScenesPerBatch"]
    dataloader_cache_capacity = specs["DataLoaderSpecs"]["CacheCapacity"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    test_dataset = utils.data.InterceptDataset(data_source, test_split, dataloader_cache_capacity)
    
    # get dataloader
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    return test_dataloader


def save_result(test_dataloader, pcd, indices, specs, extend_info=None):
    save_dir = specs["SaveDir"]
    filename_patten = specs["FileNamePatten"]
    # 将udf数据拆分开，并且转移到cpu
    pcd_np = pcd.cpu().detach().numpy()

    filename_list = [test_dataloader.dataset.pcd1files[index] for index in indices]
    for index, filename_abs in enumerate(filename_list):
        filename_info = os.path.split(filename_abs)
        # get the pure filename and the category of the data
        filename_relative = re.match(filename_patten, filename_info[-1]).group()  # scene1.1001_view0
        category = filename_info[-2]  # IBSNet_scan512/scene1

        # the real directory is save_dir/category
        save_path = os.path.join(save_dir, category)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        # append the extend info to the filename
        if extend_info is not None:
            filename_relative += "_{}".format(extend_info)
        filename_final = "{}.ply".format(filename_relative)
        absolute_dir = os.path.join(save_path, filename_final)

        o3d.io.write_point_cloud(absolute_dir, get_pcd_from_np(pcd_np[index]))


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def test(IBPCDCNet, test_dataloader, specs, model):
    device = specs["Device"]
    loss_weight_out_gt = specs["LossSpecs"]["WeightOutGt"]
    loss_weight_out_out = specs["LossSpecs"]["WeightOutOut"]
    test_result_dir = specs["TestResult"]
    test_split = specs["TestSplit"]
    visualize = specs["Visualize"]
    save = specs["Save"]

    loss_cd_pcd1_path1 = networks.loss.cdModule()
    loss_cd_pcd2_path1 = networks.loss.cdModule()
    loss_cd_pcd1_path2 = networks.loss.cdModule()
    loss_cd_pcd2_path2 = networks.loss.cdModule()
    loss_cd_pcd1 = networks.loss.cdModule()
    loss_cd_pcd2 = networks.loss.cdModule()

    with torch.no_grad():
        test_total_loss = 0
        for IBS, pcd1, pcd2, pcd1gt, pcd2gt, idx in test_dataloader:
            IBS.requires_grad = False
            pcd1.requires_grad = False
            pcd2.requires_grad = False
            pcd1gt.requires_grad = False
            pcd2gt.requires_grad = False

            pcd1 = pcd1.to(device)
            pcd2 = pcd2.to(device)
            IBS = IBS.to(device)

            # visualize_data1(pcd1, pcd2, xyz, udf_gt1, udf_gt2)
            pcd1_path1, pcd2_path1, pcd1_path2, pcd2_path2 = IBPCDCNet(pcd1, pcd2, IBS)

            # loss between out and groundtruth
            loss_pcd1_path1_gt = loss_cd_pcd1_path1(pcd1_path1, pcd1gt.to(device))
            loss_pcd2_path1_gt = loss_cd_pcd2_path1(pcd2_path1, pcd2gt.to(device))
            loss_pcd1_path2_gt = loss_cd_pcd1_path2(pcd1_path2, pcd1gt.to(device))
            loss_pcd2_path2_gt = loss_cd_pcd2_path2(pcd2_path2, pcd2gt.to(device))
            # loss between two path
            loss_pcd1_path1_path2 = loss_cd_pcd1(pcd1_path1, pcd1_path2)
            loss_pcd2_path1_path2 = loss_cd_pcd2(pcd2_path1, pcd2_path2)

            batch_loss = loss_weight_out_gt * (
                        loss_pcd1_path1_gt + loss_pcd2_path1_gt + loss_pcd1_path2_gt + loss_pcd2_path2_gt) + \
                         loss_weight_out_out * (loss_pcd1_path1_path2 + loss_pcd2_path1_path2)

            test_total_loss += batch_loss.item()

            if save:
                save_result(test_dataloader, pcd1_path1, idx, specs, "{}_{}".format("pcd1", "path1"))
                save_result(test_dataloader, pcd2_path1, idx, specs, "{}_{}".format("pcd2", "path1"))
                save_result(test_dataloader, pcd1_path2, idx, specs, "{}_{}".format("pcd1", "path2"))
                save_result(test_dataloader, pcd2_path2, idx, specs, "{}_{}".format("pcd2", "path2"))
            if visualize:
                visualize_data(pcd1_path1, pcd2_path1, pcd1_path2, pcd2_path2, specs)

            print("handled a batch, idx: {}", idx)
        test_avrg_loss = test_total_loss / test_dataloader.__len__()
        print(' test_avrg_loss: {}\n'.format(test_avrg_loss))

        # 写入测试结果
        test_split_ = test_split.replace("/", "-").replace("\\", "-")
        model_ = model.replace("/", "-").replace("\\", "-")
        test_result_filename = os.path.join(test_result_dir, "{}+{}.txt".format(test_split_, model_))
        with open(test_result_filename, 'w') as f:
            f.write("test_split: {}\n".format(test_split))
            f.write("model: {}\n".format(model))
            f.write("avrg_loss: {}\n".format(test_avrg_loss))


def main_function(experiment_config_file, model_path):
    specs = ws.load_experiment_specifications(experiment_config_file)
    device = specs["Device"]
    print("test device: {}".format(device))

    test_dataloader = get_dataloader(specs)
    print("init dataloader succeed")

    # 读取模型
    IBPCDCNet = torch.load(model_path, map_location="cuda:{}".format(device))
    print("load trained model succeed")

    # 测试并返回loss
    test(IBPCDCNet, test_dataloader, specs, model_path)


if __name__ == '__main__':
    print("-------------------begin test-----------------")

    import argparse
    arg_parser = argparse.ArgumentParser(description="Train a IBS Net")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_test.json",
        required=False,
        help="The experiment config file."
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="trained_models/train_2023-11-15_20-05-11/epoch_195.pth",
        required=False,
        help="The network para"
    )

    args = arg_parser.parse_args()

    print("specs file: {}, model path: {}".format(args.experiment_config_file, args.model))

    main_function(args.experiment_config_file, args.model)
