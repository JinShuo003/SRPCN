import sys
import os

sys.path.insert(0, os.path.abspath("."))

import os.path

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from datetime import datetime, timedelta
import argparse
import time

from models.SRPCN import SRPCN
from models.pn2_utils import fps_subsample
from utils import path_utils
from utils.loss import cd_loss_L1, cd_loss_L1_single, medial_axis_surface_loss, medial_axis_interaction_loss, \
    ibs_angle_loss, emd_loss
from utils.train_utils import *
from dataset import dataset_INTE


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
        pcds_pred = network(pcd_partial)

        pcd_gt = pcd_gt.to(device)
        center = center.to(device)
        radius = radius.to(device)
        direction = direction.to(device)

        Pc, P1, P2, P3 = pcds_pred

        pcd_gt_2 = fps_subsample(pcd_gt, P2.shape[1])
        pcd_gt_1 = fps_subsample(pcd_gt_2, P1.shape[1])
        pcd_gt_c = fps_subsample(pcd_gt_1, Pc.shape[1])

        cdc = cd_loss_L1(Pc, pcd_gt_c)
        cd1 = cd_loss_L1(P1, pcd_gt_1)
        cd2 = cd_loss_L1(P2, pcd_gt_2)
        cd3 = cd_loss_L1(P3, pcd_gt)

        partial_matching = cd_loss_L1_single(pcd_partial, P3)

        loss_medial_axis_surface = medial_axis_surface_loss(center, radius, P3)
        loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, P3)
        loss_ibs_angle, intersect_num = ibs_angle_loss(center, radius, direction, P3)

        loss_total = (cdc + cd1 + cd2 + cd3 + partial_matching +
                      mads_loss_weight * loss_medial_axis_surface +
                      madi_loss_weight * loss_medial_axis_interaction +
                      ibsa_loss_weight * loss_ibs_angle)

        train_total_loss_dense += cd3.item()
        train_total_loss_sub_dense += cd2.item()
        train_total_loss_coarse += cdc.item()
        train_total_loss_medial_axis_surface += loss_medial_axis_surface.item()
        train_total_loss_medial_axis_interaction += loss_medial_axis_interaction.item()
        train_total_loss_ibs_angle += loss_ibs_angle.item()
        train_total_intersect_num += intersect_num.item()

        loss_total.backward()
        optimizer.step()

    lr_schedule.step()

    record_loss_info(specs, "train_loss_dense", train_total_loss_dense / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info(specs, "train_loss_sub_dense", train_total_loss_sub_dense / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info(specs, "train_loss_coarse", train_total_loss_coarse / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info(specs, "train_loss_medial_axis_surface",
                     train_total_loss_medial_axis_surface / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_medial_axis_interaction",
                     train_total_loss_medial_axis_interaction / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_ibs_angle", train_total_loss_ibs_angle / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info(specs, "train_intersect_num", train_total_intersect_num / train_dataloader.__len__(), epoch,
                     tensorboard_writer)


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

            pcds_pred = network(pcd_partial)

            pcd_gt = pcd_gt.to(device)
            center = center.to(device)
            radius = radius.to(device)
            direction = direction.to(device)

            Pc, P1, P2, P3 = pcds_pred

            pcd_gt_2 = fps_subsample(pcd_gt, P2.shape[1])
            pcd_gt_1 = fps_subsample(pcd_gt_2, P1.shape[1])
            pcd_gt_c = fps_subsample(pcd_gt_1, Pc.shape[1])

            cdc = cd_loss_L1(Pc, pcd_gt_c)
            cd1 = cd_loss_L1(P1, pcd_gt_1)
            cd2 = cd_loss_L1(P2, pcd_gt_2)
            cd3 = cd_loss_L1(P3, pcd_gt)

            partial_matching = cd_loss_L1_single(pcd_partial, P3)

            loss_medial_axis_surface = medial_axis_surface_loss(center, radius, P3)
            loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, P3)
            loss_ibs_angle, intersect_num = ibs_angle_loss(center, radius, direction, P3)
            loss_emd = emd_loss(P3, pcd_gt)

            test_total_dense += cd3.item()
            test_total_sub_dense += cd2.item()
            test_total_coarse += cdc.item()
            test_total_medial_axis_surface += loss_medial_axis_surface.item()
            test_total_medial_axis_interaction += loss_medial_axis_interaction.item()
            test_total_ibs_angle += loss_ibs_angle.item()
            test_total_intersect_num += intersect_num.item()
            test_total_emd += loss_emd.item()

        test_avrg_dense = test_total_dense / test_dataloader.__len__()
        record_loss_info(specs, "test_loss_dense", test_total_dense / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
        record_loss_info(specs, "test_loss_sub_dense", test_total_sub_dense / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
        record_loss_info(specs, "test_loss_coarse", test_total_coarse / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
        record_loss_info(specs, "test_loss_medial_axis_surface",
                         test_total_medial_axis_surface / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_medial_axis_interaction",
                         test_total_medial_axis_interaction / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_ibs_angle", test_total_ibs_angle / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
        record_loss_info(specs, "test_intersect_num", test_total_intersect_num / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
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

    train_loader, test_loader = get_dataloader(dataset_INTE.INTEDataset, specs)
    checkpoint = get_checkpoint(specs)
    network = get_network(specs, SRPCN, checkpoint)
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
        best_cd, best_epoch = test(network, test_loader, lr_scheduler, optimizer, epoch, specs, tensorboard_writer,
                                   best_cd, best_epoch)
        time_end_test = time.time()
        logger.info("use {} to test".format(time_end_test - time_begin_test))

    tensorboard_writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train IBPCDC")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/INTE/train/specs_train_RSAPCN_INTE.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
