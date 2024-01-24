import sys
from datetime import datetime, timedelta

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch.utils.data as data_utils
import json
import re
import argparse
import time
import open3d as o3d
import numpy as np
import shutil

from models.PCN import PCN
from utils.metric import *

from utils.geometry_utils import get_pcd_from_np
from utils import log_utils, path_utils, statistics_utils
from dataset import data_INTE

logger = None


def get_dataloader(specs):
    data_source = specs.get("DataSource")
    test_split_file = specs.get("TestSplit")
    batch_size = specs.get("BatchSize")
    num_data_loader_threads = specs.get("DataLoaderThreads")

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    test_dataset = data_INTE.INTEDataset(data_source, test_split)
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

    pcd_np = pcd.cpu().detach().numpy()

    filename_list = [test_dataloader.dataset.pcd_partial_filenames[index] for index in indices]
    for index, filename_abs in enumerate(filename_list):
        # [dataset, category, filename], example:[MVP, scene1, scene1.1000_view0_0.ply]
        dataset, category, filename = filename_abs.split('/')
        filename = re.match(filename_patten, filename).group()
        scene = re.match(scene_patten, filename).group()

        # normalize parameters
        normalize_para_filename = "{}_{}.txt".format(scene, re.findall(r'\d+', filename)[-1])
        normalize_para_path = os.path.join(normalize_para_dir, category, normalize_para_filename)
        translate, scale = get_normalize_para(normalize_para_path)

        # the real directory is save_dir/dataset/category
        save_path = os.path.join(save_dir, dataset, category)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # get final filename
        filename_final = "{}.ply".format(filename)
        absolute_path = os.path.join(save_path, filename_final)
        pcd = get_pcd_from_np(pcd_np[index])

        # transform to origin coordinate
        pcd.scale(scale, np.array([0, 0, 0]))
        pcd.translate(translate)

        o3d.io.write_point_cloud(absolute_path, pcd)


def create_zip(dataset: str = None):
    save_dir = specs.get("ResultSaveDir")

    output_archive = os.path.join(save_dir, dataset)

    shutil.make_archive(output_archive, 'zip', output_archive)


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
    for i in range(1, 10):
        category = "scene{}".format(i)
        dist_dict[category]["avrg_dist"] = dist_dict[category]["dist_total"] / dist_dict[category]["num"]
        dist_total += dist_dict[category]["dist_total"]
        num += dist_dict[category]["num"]
    dist_dict["avrg_dist"] = dist_total / num


def test(network, test_dataloader, specs):
    device = specs.get("Device")

    dist_dict = {
        "cd_l1": {},
        "cd_l2": {},
        "emd": {},
        "fscore": {},
        "mad_s": {},
        "mad_i": {},
        "ibs_a": {}
    }
    network.eval()
    with torch.no_grad():
        for data, idx in test_dataloader:
            center, radius, direction, pcd_partial, pcd_gt = data
            pcd_partial = pcd_partial.to(device)
            pcd_gt = pcd_gt.to(device)
            center = center.to(device)
            radius = radius.to(device)
            direction = direction.to(device)
            
            coarse_pred, dense_pred = network(pcd_partial)

            cd_l1 = l1_cd(dense_pred, pcd_gt)
            cd_l2 = l2_cd(dense_pred, pcd_gt)
            emd_ = emd(dense_pred, pcd_gt)
            fscore = f_score(dense_pred, pcd_gt)
            mad_s = medial_axis_surface_dist(center, radius, dense_pred)
            mad_i = medial_axis_interaction_dist(center, radius, dense_pred)
            ibs_a = ibs_angle_dist(center, dense_pred, direction)

            update_loss_dict(dist_dict, cd_l1.detach().cpu().numpy(), test_dataloader, idx, "cd_l1")
            update_loss_dict(dist_dict, cd_l2.detach().cpu().numpy(), test_dataloader, idx, "cd_l2")
            update_loss_dict(dist_dict, emd_.detach().cpu().numpy(), test_dataloader, idx, "emd")
            update_loss_dict(dist_dict, fscore.detach().cpu().numpy(), test_dataloader, idx, "fscore")
            update_loss_dict(dist_dict, mad_s.detach().cpu().numpy(), test_dataloader, idx, "mad_s")
            update_loss_dict(dist_dict, mad_i.detach().cpu().numpy(), test_dataloader, idx, "mad_i")
            update_loss_dict(dist_dict, ibs_a.detach().cpu().numpy(), test_dataloader, idx, "ibs_a")

            save_result(test_dataloader, dense_pred, idx, specs)
            logger.info("saved {} pcds".format(idx.shape[0]))

        cal_avrg_dist(dist_dict, "cd_l1")
        cal_avrg_dist(dist_dict, "cd_l2")
        cal_avrg_dist(dist_dict, "emd")
        cal_avrg_dist(dist_dict, "fscore")
        cal_avrg_dist(dist_dict, "mad_s")
        cal_avrg_dist(dist_dict, "mad_i")
        cal_avrg_dist(dist_dict, "ibs_a")

        logger.info("dist result: \n{}".format(json.dumps(dist_dict, sort_keys=False, indent=4)))
        csv_file_dir = os.path.join(specs.get("LogDir"), specs.get("TAG"))
        csv_file_path = os.path.join(csv_file_dir, "evaluate_result.csv")
        statistics_utils.write_avrg_csv_file(csv_file_path, dist_dict, "INTE")


def main_function(specs, model_path):
    device = specs.get("Device")
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(specs)
    logger.info("init dataloader succeed")

    model = PCN(num_dense=2048, device=device).to(device)
    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    model.load_state_dict(checkpoint["model"])
    logger.info("load trained model succeed, epoch: {}".format(checkpoint["epoch"]))

    time_begin_test = time.time()
    test(model, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))

    time_begin_zip = time.time()
    create_zip(dataset="INTE_norm")
    time_end_zip = time.time()
    logger.info("use {} to zip".format(time_end_zip - time_begin_zip))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/INTE/test/specs_test_PCN_INTE.json",
        required=False,
        help="The experiment config file."
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="model_paras/PCN_INTE/epoch_58.pth",
        required=False,
        help="The network para"
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = log_utils.get_test_logger(specs)
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("test split: {}".format(specs.get("TestSplit")))
    logger.info("specs file: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))
    logger.info("model: {}".format(args.model))

    main_function(specs, args.model)
