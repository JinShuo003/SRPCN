import sys
sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import logging
import os.path

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import json
import open3d as o3d
import numpy as np
import re

import networks.loss
from networks.model_transformer_TopNet_2obj_ibs import *

import utils.data
import utils.workspace as ws
from utils.geometry_utils import get_pcd_from_np


def visualize_data(pcd1, pcd2, specs):
    # 将udf数据拆分开，并且转移到cpu
    pcd1 = pcd1.cpu().detach().numpy()
    pcd2 = pcd2.pcd2().detach().numpy()

    for i in range(pcd1.shape[0]):
        pcd1_ = get_pcd_from_np(pcd1[i])
        pcd2_ = get_pcd_from_np(pcd2[i])

        pcd1_.paint_uniform_color([1, 0, 0])
        pcd2_.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([pcd1_, pcd2_])


def get_dataloader(specs):
    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]
    scene_per_batch = specs["ScenesPerBatch"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    test_dataset = utils.data.IntersectDataset(data_source, test_split)
    
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
    test_result_dir = specs["TestResult"]
    test_split = specs["TestSplit"]
    visualize = specs["Visualize"]
    save = specs["Save"]

    loss_emd = networks.loss.emdModule()
    loss_cd = networks.loss.cdModule()

    with torch.no_grad():
        test_total_loss_emd = 0
        test_total_loss_cd = 0
        for IBS, pcd1_partial, pcd2_partial, pcd1_gt, pcd2_gt, idx in test_dataloader:
            IBS.requires_grad = False
            pcd1_partial.requires_grad = False
            pcd2_partial.requires_grad = False
            pcd1_gt.requires_grad = False
            pcd2_gt.requires_grad = False

            IBS = IBS.to(device)
            pcd1_partial = pcd1_partial.to(device)
            pcd2_partial = pcd2_partial.to(device)
            pcd1_out, pcd2_out = IBPCDCNet(pcd1_partial, pcd2_partial, IBS)

            pcd1_gt = pcd1_gt.to(device)
            pcd2_gt = pcd2_gt.to(device)
            loss_emd_pcd1 = torch.mean(loss_emd(pcd1_gt, pcd1_out)[0])
            loss_emd_pcd2 = torch.mean(loss_emd(pcd2_gt, pcd2_out)[0])
            # loss_cd_pcd1 = loss_cd(pcd1_out, pcd1_gt)
            # loss_cd_pcd2 = loss_cd(pcd2_out, pcd2_gt)

            batch_loss_emd = loss_emd_pcd1 + loss_emd_pcd2
            # batch_loss_cd = loss_cd_pcd1 + loss_cd_pcd2

            test_total_loss_emd += batch_loss_emd.item()
            # test_total_loss_cd += batch_loss_cd.item()

            if save:
                save_result(test_dataloader, pcd1_out, idx, specs, "{}".format("0"))
                save_result(test_dataloader, pcd2_out, idx, specs, "{}".format("1"))
            if visualize:
                visualize_data(pcd1_out, pcd2_out, specs)

        test_avrg_loss_emd = test_total_loss_emd / test_dataloader.__len__()
        print(' test_avrg_loss_emd: {}\n'.format(test_avrg_loss_emd))
        # test_avrg_loss_cd = test_total_loss_cd / test_dataloader.__len__()
        # print(' test_avrg_loss_cd: {}\n'.format(test_avrg_loss_cd))

        # 写入测试结果
        test_split_ = test_split.replace("/", "-").replace("\\", "-")
        model_ = model.replace("/", "-").replace("\\", "-")
        test_result_filename = os.path.join(test_result_dir, "{}+{}.txt".format(test_split_, model_))
        with open(test_result_filename, 'w') as f:
            f.write("test_split: {}\n".format(test_split))
            f.write("model: {}\n".format(model))
            f.write("avrg_loss_emd: {}\n".format(test_avrg_loss_emd))
            # f.write("avrg_loss_cd: {}\n".format(test_avrg_loss_cd))


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
        default="trained_models/train_2023-12-23_18-19-46/epoch_195.pth",
        required=False,
        help="The network para"
    )

    args = arg_parser.parse_args()

    print("specs file: {}, model path: {}".format(args.experiment_config_file, args.model))

    main_function(args.experiment_config_file, args.model)
