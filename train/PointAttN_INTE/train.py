import sys
import os

sys.path.insert(0, os.path.abspath("."))

import os.path
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from datetime import datetime, timedelta
import json
import argparse
import time
import torch

from models.PointAttN import PointAttN
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from utils import path_utils
from utils.loss import cd_loss_L1, medial_axis_surface_loss, medial_axis_interaction_loss, ibs_angle_loss, emd_loss
from utils.train_utils import *
from dataset import data_INTE


def train(network, train_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")

    network.train()
    logger.info("")
    logger.info('epoch: {}, learning rate: {}'.format(epoch, optimizer.param_groups[0]["lr"]))

    mads_loss_weight = get_loss_weight(specs.get("MADSLossOptions"), epoch)
    madi_loss_weight = get_loss_weight(specs.get("MADILossOptions"), epoch)
    ibsa_loss_weight = get_loss_weight(specs.get("IBSALossOptions"), epoch)

    logger.info("mads_loss_weight: {}".format(mads_loss_weight))
    logger.info("madi_loss_weight: {}".format(madi_loss_weight))
    logger.info("ibsa_loss_weight: {}".format(ibsa_loss_weight))

    train_total_loss_dense = 0
    train_total_loss_sub_dense = 0
    train_total_loss_coarse = 0
    train_total_loss_medial_axis_surface = 0
    train_total_loss_medial_axis_interaction = 0
    train_total_loss_ibs_angle = 0
    train_total_intersect_num = 0
    
    for data, idx in train_dataloader:
        center, radius, direction, pcd_partial, pcd_gt = data
        optimizer.zero_grad()

        pcd_partial = pcd_partial.to(device)
        pcd_pred_coarse, pcd_pred_sub_dense, pcd_pred_dense = network(pcd_partial)

        pcd_gt = pcd_gt.to(device)
        center = center.to(device)
        radius = radius.to(device)
        direction = direction.to(device)

        loss_dense = cd_loss_L1(pcd_pred_dense, pcd_gt)
        gt_sub_dense = gather_operation(pcd_gt.transpose(1, 2).contiguous(), furthest_point_sample(pcd_gt, pcd_pred_sub_dense.shape[1])).transpose(1, 2).contiguous()
        loss_sub_dense = cd_loss_L1(pcd_pred_sub_dense, gt_sub_dense)
        gt_coarse = gather_operation(gt_sub_dense.transpose(1, 2).contiguous(), furthest_point_sample(gt_sub_dense, pcd_pred_coarse.shape[1])).transpose(1, 2).contiguous()
        loss_coarse = cd_loss_L1(pcd_pred_dense, gt_coarse)
        loss_medial_axis_surface = medial_axis_surface_loss(center, radius, pcd_pred_dense)
        loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, pcd_pred_dense)
        loss_ibs_angle, intersect_num = ibs_angle_loss(center, radius, direction, pcd_pred_dense)

        loss_total = loss_dense + loss_sub_dense + loss_coarse + \
                    mads_loss_weight * loss_medial_axis_surface + \
                    madi_loss_weight * loss_medial_axis_interaction + \
                    ibsa_loss_weight * loss_ibs_angle
                    
        train_total_loss_dense += loss_dense.item()
        train_total_loss_sub_dense += loss_sub_dense.item()
        train_total_loss_coarse += loss_coarse.item()
        train_total_loss_medial_axis_surface += loss_medial_axis_surface.item()
        train_total_loss_medial_axis_interaction += loss_medial_axis_interaction.item()
        train_total_loss_ibs_angle += loss_ibs_angle.item()
        train_total_intersect_num += intersect_num.item()

        loss_total.backward()
        optimizer.step()

    lr_schedule.step()

    record_loss_info(specs, "train_loss_dense", train_total_loss_dense / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_sub_dense", train_total_loss_sub_dense / train_dataloader.__len__(), epoch,tensorboard_writer)
    record_loss_info(specs, "train_loss_coarse", train_total_loss_coarse / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_medial_axis_surface", train_total_loss_medial_axis_surface / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_medial_axis_interaction", train_total_loss_medial_axis_interaction / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_ibs_angle", train_total_loss_ibs_angle / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_intersect_num", train_total_intersect_num / train_dataloader.__len__(), epoch, tensorboard_writer)


def test(network, test_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer, best_cd, best_epoch):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")

    network.eval()
    with torch.no_grad():
        test_total_dense = 0
        test_total_sub_dense = 0
        test_total_coarse = 0
        test_total_medial_axis_surface = 0
        test_total_medial_axis_interaction = 0
        test_total_ibs_angle = 0
        test_total_intersect_num = 0
        test_total_emd = 0
        for data, idx in test_dataloader:
            center, radius, direction, pcd_partial, pcd_gt = data
            pcd_partial = pcd_partial.to(device)

            pcd_pred_coarse, pcd_pred_sub_dense, pcd_pred_dense = network(pcd_partial)

            pcd_gt = pcd_gt.to(device)
            center = center.to(device)
           
            radius = radius.to(device)
            direction = direction.to(device)

            loss_dense = cd_loss_L1(pcd_pred_dense, pcd_gt)
            loss_sub_dense = cd_loss_L1(pcd_pred_sub_dense, pcd_gt)
            loss_coarse = cd_loss_L1(pcd_pred_coarse, pcd_gt)
            loss_medial_axis_surface = medial_axis_surface_loss(center, radius, pcd_pred_dense)
            loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, pcd_pred_dense)
            loss_ibs_angle, intersect_num = ibs_angle_loss(center, radius, direction, pcd_pred_dense)
            loss_emd = emd_loss(pcd_pred_dense, pcd_gt)

            test_total_dense += loss_dense.item()
            test_total_sub_dense += loss_sub_dense.item()
            test_total_coarse += loss_coarse.item()
            test_total_medial_axis_surface += loss_medial_axis_surface.item()
            test_total_medial_axis_interaction += loss_medial_axis_interaction.item()
            test_total_ibs_angle += loss_ibs_angle.item()
            test_total_intersect_num += intersect_num.item()
            test_total_emd += loss_emd.item()

        test_avrg_dense = test_total_dense / test_dataloader.__len__()
        record_loss_info(specs, "test_loss_dense", test_total_dense / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_sub_dense", test_total_sub_dense / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_coarse", test_total_coarse / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_medial_axis_surface", test_total_medial_axis_surface / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_medial_axis_interaction", test_total_medial_axis_interaction / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_ibs_angle", test_total_ibs_angle / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_intersect_num", test_total_intersect_num / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_emd", test_total_emd / test_dataloader.__len__(), epoch, tensorboard_writer)

        if test_avrg_dense < best_cd:
            best_epoch = epoch
            best_cd = test_avrg_dense
            logger.info('current best epoch: {}, cd: {}'.format(best_epoch, best_cd))
        save_model(specs, network, lr_schedule, optimizer, epoch)

        return best_cd, best_epoch


def main_function(specs):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    epoch_num = specs.get("TrainOptions").get("NumEpochs")
    continue_train = specs.get("TrainOptions").get("ContinueTrain")

    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))

    logger.info("current network TAG: {}".format(specs.get("TAG")))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("There are {} epochs in total".format(epoch_num))

    train_loader, test_loader = get_dataloader(data_INTE.INTEDataset, specs)
    checkpoint = get_checkpoint(specs)
    network = get_network(specs, PointAttN, checkpoint)
    optimizer = get_optimizer(specs, network, checkpoint)
    lr_scheduler_class, kwargs = get_lr_scheduler_info(specs)
    lr_scheduler = get_lr_scheduler(specs, optimizer, checkpoint, lr_scheduler_class, **kwargs)
    tensorboard_writer = get_tensorboard_writer(specs)

    best_cd = 1e8
    best_epoch = -1
    epoch_begin = 0
    if continue_train:
        last_epoch = specs.get("TrainOptions").get("ContinueFromEpoch")
        epoch_begin = last_epoch + 1
        logger.info("continue train from epoch {}".format(epoch_begin))
    for epoch in range(epoch_begin, epoch_num + 1):
        time_begin_train = time.time()
        train(network, train_loader, lr_scheduler, optimizer, epoch, specs, tensorboard_writer)
        time_end_train = time.time()
        logger.info("use {} to train".format(time_end_train - time_begin_train))

        time_begin_test = time.time()
        best_cd, best_epoch = test(network, test_loader, lr_scheduler, optimizer, epoch, specs, tensorboard_writer, best_cd, best_epoch)
        time_end_test = time.time()
        logger.info("use {} to test".format(time_end_test - time_begin_test))

    tensorboard_writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train IBPCDC")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/INTE/train/specs_train_PointAttN_INTE.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
