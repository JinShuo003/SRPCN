import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path

import torch.utils.data as data_utils
import json
import open3d as o3d
import numpy as np
import re
import argparse
import time

from networks.loss import *

from utils.geometry_utils import get_pcd_from_np
from utils import log_utils, path_utils
from dataset import data_normalize

logger = None


def get_dataloader(specs):
    data_source = specs.get("DataSource")
    test_split_file = specs.get("TestSplit")
    batch_size = specs.get("BatchSize")
    num_data_loader_threads = specs.get("DataLoaderThreads")

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    test_dataset = data_normalize.InteractionDataset(data_source, test_split)

    # get dataloader
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    return test_dataloader


def get_normalize_para(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        data = list(map(float, content.split(",")))
        translate = data[0:3]
        scale = data[-1]
    return translate, scale


def save_result(test_dataloader, pcd, indices, specs):
    save_dir = specs.get("ResultSaveDir")
    normalize_para_dir = specs.get("NormalizeParaDir")

    filename_patten = specs.get("FileNamePatten")
    scene_patten = specs.get("ScenePatten")

    # 将udf数据拆分开，并且转移到cpu
    pcd_np = pcd.cpu().detach().numpy()

    filename_list = [test_dataloader.dataset.pcd_partial_filenames[index] for index in indices]
    for index, filename_abs in enumerate(filename_list):
        # [pcd_partial_2048, scene1, scene1.1000_view0_0]
        filename_info = filename_abs.split('/')
        filename = re.match(filename_patten, filename_info[2]).group()
        scene = re.match(scene_patten, filename_info[2]).group()
        category = filename_info[1]

        # normalize parameters
        normalize_para_filename = "{}_{}.txt".format(scene, re.findall(r'\d+', filename)[-1])
        normalize_para_path = os.path.join(normalize_para_dir, category, normalize_para_filename)
        translate, scale = get_normalize_para(normalize_para_path)

        # the real directory is save_dir/category
        save_path = os.path.join(save_dir, filename_info[0], category)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # append the extend info to the filename
        filename_final = "{}.ply".format(filename)
        absolute_dir = os.path.join(save_path, filename_final)
        pcd = get_pcd_from_np(pcd_np[index])

        # transform to origin coordinate
        pcd.scale(scale, np.array([0, 0, 0]))
        pcd.translate(translate)

        o3d.io.write_point_cloud(absolute_dir, pcd)


def test(IBPCDCNet, test_dataloader, specs):
    device = specs.get("Device")

    with torch.no_grad():
        test_total_loss_emd = 0
        test_total_loss_cd = 0
        for IBS, pcd_partial, pcd_gt, idx in test_dataloader:
            pcd_partial = pcd_partial.to(device)
            pcd_out = IBPCDCNet(pcd_partial)

            pcd_gt = pcd_gt.to(device)
            loss_emd_pcd = emd_loss(pcd_out, pcd_gt)
            loss_cd_pcd = cd_loss_L1(pcd_out, pcd_gt)

            batch_loss_emd = loss_emd_pcd
            batch_loss_cd = loss_cd_pcd

            test_total_loss_emd += batch_loss_emd.item()
            test_total_loss_cd += batch_loss_cd.item()

            save_result(test_dataloader, pcd_out, idx, specs)

        test_avrg_loss_emd = test_total_loss_emd / test_dataloader.__len__()
        logger.info("test_avrg_loss_emd: {}".format(test_avrg_loss_emd))
        test_avrg_loss_cd = test_total_loss_cd / test_dataloader.__len__()
        logger.info("test_avrg_loss_cd: {}".format(test_avrg_loss_cd))


def main_function(specs, model_path):
    device = specs.get("Device")
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(specs)
    logger.info("init dataloader succeed")

    IBPCDCNet = torch.load(model_path, map_location="cuda:{}".format(device))
    logger.info("load trained model succeed")

    time_begin_test = time.time()
    test(IBPCDCNet, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_test_Transformer_TopNet_1obj_no_ibs.json",
        required=False,
        help="The experiment config file."
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="trained_models/Transformer_TopNet_1obj_no_ibs/epoch_60.pth",
        required=False,
        help="The network para"
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = log_utils.get_test_logger(specs)

    logger.info("test split: {}".format(specs.get("TestSplit")))
    logger.info("specs file: {}".format(args.experiment_config_file))
    logger.info("model: {}".format(args.model))

    main_function(specs, args.model)
