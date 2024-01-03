import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path

import torch.utils.data as data_utils
import json
import re
import argparse
import time
import numpy as np
import open3d as o3d
import shutil

from networks.model_PCN import *
from utils.loss import *
from utils.metric import *

from utils.geometry_utils import get_pcd_from_np
from utils import log_utils, path_utils
from dataset import dataset_MVP

logger = None


def get_dataloader(specs):
    data_source = specs.get("DataSource")
    test_split_file = specs.get("TestSplit")
    batch_size = specs.get("BatchSize")
    num_data_loader_threads = specs.get("DataLoaderThreads")

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    test_dataset = dataset_MVP.PcdDataset(data_source, test_split)
    logger.info("length of test_dataset: {}".format(test_dataset.__len__()))

    # get dataloader
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logger.info("length of test_dataloader: {}".format(test_dataloader.__len__()))

    return test_dataloader


def save_result(test_dataloader, pcd, indices, specs):
    save_dir = specs.get("ResultSaveDir")

    # 将udf数据拆分开，并且转移到cpu
    pcd_np = pcd.cpu().detach().numpy()

    filename_list = [test_dataloader.dataset.pcd_partial_filenames[index] for index in indices]
    for index, filename_abs in enumerate(filename_list):
        # [dataset, category, filename], example:[MVP, scene1, scene1.1000_view0_0.ply]
        dataset, category, filename = filename_abs.split('/')

        save_path = os.path.join(save_dir, dataset, category)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        pcd_save_absolute_path = os.path.join(save_path, filename)

        o3d.io.write_point_cloud(pcd_save_absolute_path, get_pcd_from_np(pcd_np[index]))


def create_zip(dataset="MVP"):
    save_dir = specs.get("ResultSaveDir")

    output_archive = os.path.join(save_dir, dataset)

    shutil.make_archive(output_archive, 'zip', save_dir)


def update_loss_dict(dist_dict_total: dict, dist, test_dataloader, indices, tag: str):
    assert dist.shape[0] == indices.shape[0]
    assert tag in dist_dict_total.keys()

    dist_dict = dist_dict_total.get(tag)
    filename_list = [test_dataloader.dataset.pcd_gt_filenames[index] for index in indices]
    for idx, filename in enumerate(filename_list):
        category = "scene{}".format(str(int(re.findall(r'\d+', filename)[0])))
        if category not in dist_dict:
            dist_dict[category] = {
                "dist_total": 0,
                "num": 0
            }
        dist_dict[category]["dist_total"] += dist[idx]
        dist_dict[category]["num"] += 1


def cal_avrg_dist(dist_dict_total: dict, tag: str):
    dist_dict = dist_dict_total.get(tag)
    dist_total = 0
    num = 0
    for i in range(1, 17):
        category = "scene{}".format(i)
        dist_dict[category]["avrg_dist"] = dist_dict[category]["dist_total"] / dist_dict[category]["num"]
        dist_total += dist_dict[category]["dist_total"]
        num += dist_dict[category]["num"]
    dist_dict["avrg_dist"] = dist_total/num


def test(network, test_dataloader, specs):
    device = specs.get("Device")

    dist_dict = {
        "cd_l1": {},
        "cd_l2": {},
        "emd": {},
        "fscore": {}
    }
    network.eval()
    with torch.no_grad():
        for pcd_partial, pcd_gt, idx in test_dataloader:
            pcd_partial = pcd_partial.to(device)
            pcd_gt = pcd_gt.to(device)

            coarse_pred, dense_pred = network(pcd_partial)

            cd_l1 = l1_cd(dense_pred, pcd_gt)
            cd_l2 = l2_cd(dense_pred, pcd_gt)
            emd_ = emd(dense_pred, pcd_gt)
            fscore = f_score(dense_pred, pcd_gt)

            update_loss_dict(dist_dict, cd_l1.detach().cpu().numpy(), test_dataloader, idx, "cd_l1")
            update_loss_dict(dist_dict, cd_l2.detach().cpu().numpy(), test_dataloader, idx, "cd_l2")
            update_loss_dict(dist_dict, emd_.detach().cpu().numpy(), test_dataloader, idx, "emd")
            update_loss_dict(dist_dict, fscore.detach().cpu().numpy(), test_dataloader, idx, "fscore")

            save_result(test_dataloader, dense_pred, idx, specs)
            logger.info("saved {} pcds".format(idx.shape[0]))

        cal_avrg_dist(dist_dict, "cd_l1")
        cal_avrg_dist(dist_dict, "cd_l2")
        cal_avrg_dist(dist_dict, "emd")
        cal_avrg_dist(dist_dict, "fscore")

        logger.info("dist result: \n{}".format(json.dumps(dist_dict, sort_keys=False, indent=4)))

        

def main_function(specs, model_path):
    device = specs.get("Device")
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(specs)
    logger.info("init dataloader succeed")

    model = PCN(num_dense=2048).to(device)
    state_dict = torch.load(model_path, map_location="cuda:{}".format(device))
    model.load_state_dict(state_dict)
    logger.info("load trained model succeed")

    time_begin_test = time.time()
    test(model, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))

    time_begin_zip = time.time()
    create_zip()
    time_end_zip = time.time()
    logger.info("use {} to zip".format(time_end_zip - time_begin_zip))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_test_PCN.json",
        required=False,
        help="The experiment config file."
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="trained_models/PCN/epoch_277.pth",
        required=False,
        help="The network para"
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = log_utils.get_test_logger(specs)

    logger.info("test split: {}".format(specs.get("TestSplit")))
    logger.info("specs file: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))
    logger.info("model: {}".format(args.model))

    main_function(specs, args.model)
