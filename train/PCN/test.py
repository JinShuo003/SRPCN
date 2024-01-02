import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path

import torch.utils.data as data_utils
import json
import re
import argparse
import time

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
    test_dataset = dataset_MVP.InteractionDataset(data_source, test_split)

    # get dataloader
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    return test_dataloader


def save_result(test_dataloader, pcd, indices, specs, extend_info=None):
    save_dir = specs.get("ResultSaveDir")
    filename_patten = specs.get("FileNamePatten")
    # 将udf数据拆分开，并且转移到cpu
    pcd_np = pcd.cpu().detach().numpy()

    filename_list = [test_dataloader.dataset.pcd1files[index] for index in indices]
    for index, filename_abs in enumerate(filename_list):
        filename_info = os.path.split(filename_abs)
        # get the pure filename and the category of the data
        filename_relative = re.match(filename_patten, filename_info[-1]).group()  # scene1.1001_view0
        category = filename_info[-2]

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


def update_loss_dict(loss_dict_total: dict, loss, test_dataloader, indices, tag: str):
    assert loss.shape[0] == len(indices)
    assert tag in loss_dict_total.keys()

    loss_dict = loss_dict_total.get(tag)
    filename_list = [test_dataloader.dataset.pcd_gt_filenames[index] for index in indices]
    for idx, filename in enumerate(filename_list):
        category = "scene{}".format(re.findall(r'\d+', filename)[0] + 1)
        if category not in loss_dict:
            loss_dict[category] = {
                "total_loss": 0,
                "num": 0
            }
        loss_dict[category]["total_loss"] += loss[idx]
        loss_dict[category]["num"] += 1


def test(network, test_dataloader, specs):
    device = specs.get("Device")

    loss_dict = {
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

            cd_l1 = symmetry_cd_l1(dense_pred, pcd_gt)
            cd_l2 = symmetry_cd_l2(dense_pred, pcd_gt)
            emd = earth_movers_distance(dense_pred, pcd_gt)
            fscore = f_score(dense_pred, pcd_gt)

            update_loss_dict(loss_dict, cd_l1, test_dataloader, idx, "cd_l1")
            update_loss_dict(loss_dict, cd_l2, test_dataloader, idx, "cd_l2")
            update_loss_dict(loss_dict, emd, test_dataloader, idx, "emd")
            update_loss_dict(loss_dict, fscore, test_dataloader, idx, "fscore")


def main_function(specs, model_path):
    device = specs.get("Device")
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(specs)
    logger.info("init dataloader succeed")

    model = PCN(num_dense=2048)
    state_dict = torch.load(model_path, map_location="cuda:{}".format(device))
    model.load_state_dict(state_dict)
    logger.info("load trained model succeed")

    time_begin_test = time.time()
    test(model, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_test_PCN_TopNet_2obj_ibs.json",
        required=False,
        help="The experiment config file."
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="trained_models/PCN_TopNet_2obj_ibs/epoch_85.pth",
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
