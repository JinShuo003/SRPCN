import sys
import os

sys.path.insert(0, "/home/shuojin/IBPCDC")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import json
import argparse
import time
import os.path
from datetime import datetime, timedelta

from models.RSAPCN import RSAPCN
from utils.metric import *
from utils.test_utils import *
from utils import log_utils, path_utils
from dataset import dataset_real_world


def test(network, test_dataloader, specs):
    device = specs.get("Device")

    network.eval()
    with torch.no_grad():
        for data, idx in test_dataloader:
            filename_list = [test_dataloader.dataset.pcd_partial_filenames[i] for i in idx]
            print(filename_list)
            pcd_partial, pcd_gt = data

            pcd_partial = pcd_partial.to(device)
            pcd_gt = pcd_gt.to(device)

            Pc, P1, P2, P3 = network(pcd_partial)
            pcd_pred = P3

            save_result(filename_list, pcd_pred, specs)
            logger.info("saved {} pcds".format(idx.shape[0]))


def main_function(specs):
    device = specs.get("Device")
    model_path = specs.get("ModelPath")
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(dataset_real_world.INTEDataset, specs)
    logger.info("init dataloader succeed")

    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    model = get_network(specs, RSAPCN, checkpoint)
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
        default="configs/INTE/test/specs_test_real_world.json",
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
