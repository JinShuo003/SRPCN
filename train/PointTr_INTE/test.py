import os
import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import time
import os.path
from datetime import datetime, timedelta

from models.PoinTr import PoinTr
from utils.metric import *
from utils.test_utils import *
from utils import log_utils, path_utils, statistics_utils
from dataset import dataset_INTE


def test(network, test_dataloader, specs):
    device = specs.get("Device")

    dist_dict = {
        "cd_l1": {},
        "emd": {},
        "fscore": {},
        "mad_s": {},
        "mad_i": {},
        "ibs_a": {},
        "interact_num": {}
    }
    single_csv_data = {}
    network.eval()
    with torch.no_grad():
        for data, idx in test_dataloader:
            filename_list = [test_dataloader.dataset.pcd_partial_filenames[i] for i in idx]
            center, radius, direction, pcd_partial, pcd_gt = data

            pcd_partial = pcd_partial.to(device)
            pcd_gt = pcd_gt.to(device)
            center = center.to(device)
            radius = radius.to(device)
            direction = direction.to(device)

            coarse, dense = network(pcd_partial)

            cd_l1 = l1_cd(dense, pcd_gt)
            emd_ = emd(dense, pcd_gt)
            fscore = f_score(dense, pcd_gt)
            mad_s = medial_axis_surface_dist(center, radius, dense)
            mad_i = medial_axis_interaction_dist(center, radius, dense)
            ibs_a, interact_num = ibs_angle_dist(center, radius, direction, dense)

            update_loss_dict(dist_dict, filename_list, cd_l1.detach().cpu().numpy(), "cd_l1")
            update_loss_dict(dist_dict, filename_list, emd_.detach().cpu().numpy(), "emd")
            update_loss_dict(dist_dict, filename_list, fscore.detach().cpu().numpy(), "fscore")
            update_loss_dict(dist_dict, filename_list, mad_s.detach().cpu().numpy(), "mad_s")
            update_loss_dict(dist_dict, filename_list, mad_i.detach().cpu().numpy(), "mad_i")
            update_loss_dict(dist_dict, filename_list, ibs_a.detach().cpu().numpy(), "ibs_a")
            update_loss_dict(dist_dict, filename_list, interact_num.detach().cpu().numpy(), "interact_num")

            statistics_utils.append_csv_data(single_csv_data, filename_list, cd_l1, emd_, fscore, mad_s, mad_i, ibs_a, interact_num)

            save_result(filename_list, dense, specs)
            logger.info("saved {} pcds".format(idx.shape[0]))

        cal_avrg_dist(dist_dict)
        logger.info("dist result: \n{}".format(json.dumps(dist_dict, sort_keys=False, indent=4)))

        csv_file_dir = os.path.join(specs.get("ResultSaveDir"), specs.get("TAG"), specs.get("Dataset"))
        avrg_csv_file_path = os.path.join(csv_file_dir, "avrg_result.csv")
        statistics_utils.write_avrg_csv_file(avrg_csv_file_path, dist_dict, "INTE")

        single_csv_file_path = os.path.join(csv_file_dir, "single_result.csv")
        statistics_utils.write_single_csv_file(single_csv_file_path, single_csv_data)


def main_function(specs):
    device = specs.get("Device")
    model_path = specs.get("ModelPath")
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(dataset_INTE.INTEDataset, specs)
    logger.info("init dataloader succeed")

    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    model = get_network(specs, PoinTr, checkpoint)
    logger.info("load trained model succeed, epoch: {}".format(checkpoint["epoch"]))

    time_begin_test = time.time()
    test(model, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))

    time_begin_zip = time.time()
    create_zip(specs)
    time_end_zip = time.time()
    logger.info("use {} to zip".format(time_end_zip - time_begin_zip))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/INTE/test/specs_test_PointTr_INTE.json",
        required=False,
        help="The experiment config file."
    )
    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("test split: {}".format(specs.get("TestSplit")))
    logger.info("specs file: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))
    logger.info("model: {}".format(specs.get("ModelPath")))

    main_function(specs)
