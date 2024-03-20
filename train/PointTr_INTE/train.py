import sys
import os

sys.path.insert(0, os.path.abspath("."))

import os.path

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from datetime import datetime, timedelta
import argparse
import time
import torch.nn as nn
from torch import optim

from models.PoinTr import PoinTr, fps
from utils import path_utils
from utils.loss import cd_loss_L1, medial_axis_surface_loss, medial_axis_interaction_loss, ibs_angle_loss, emd_loss
from utils.train_utils import *
from dataset import dataset_INTE


def save_model(specs, model, epoch):
    para_save_dir = specs.get("ParaSaveDir")
    para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
    if not os.path.isdir(para_save_path):
        os.mkdir(para_save_path)

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
    }
    checkpoint_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))

    torch.save(checkpoint, checkpoint_filename)


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def get_optimizer(base_model):
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    param_groups = add_weight_decay(base_model, weight_decay=0.0005)
    optimizer = optim.AdamW(param_groups, lr=0.0001, weight_decay=0.0005)

    return optimizer


def get_lr_scheduler(base_model, optimizer, last_epoch=-1):
    warming_up_t = 0
    lr_lbmd = lambda e: max(0.9 ** ((e - warming_up_t) / 21), 0.02) if e >= warming_up_t else max(e / warming_up_t, 0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    bnm_lmbd = lambda e: max(0.9 * 0.5 ** (e / 21), 0.01)
    bnscheduler = BNMomentumScheduler(base_model, bnm_lmbd, last_epoch=last_epoch)

    scheduler = [scheduler, bnscheduler]

    return scheduler


def train(network, train_dataloader, lr_scheduler, optimizer, epoch, specs, tensorboard_writer):
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
    train_total_loss_coarse = 0
    train_total_loss_medial_axis_surface = 0
    train_total_loss_medial_axis_interaction = 0
    train_total_loss_ibs_angle = 0
    train_total_intersect_num = 0
    
    for data, idx in train_dataloader:
        center, radius, direction, pcd_partial, pcd_gt = data
        optimizer.zero_grad()

        pcd_partial = pcd_partial.to(device)
        coarse, dense = network(pcd_partial)

        pcd_gt = pcd_gt.to(device)
        center = center.to(device)
        radius = radius.to(device)
        direction = direction.to(device)

        loss_coarse = cd_loss_L1(coarse, pcd_gt)
        loss_dense = cd_loss_L1(dense, pcd_gt)
        loss_medial_axis_surface = medial_axis_surface_loss(center, radius, dense)
        loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, dense)
        loss_ibs_angle, intersect_num = ibs_angle_loss(center, radius, direction, dense)

        loss_total = loss_dense + loss_coarse + \
                    mads_loss_weight * loss_medial_axis_surface + \
                    madi_loss_weight * loss_medial_axis_interaction + \
                    ibsa_loss_weight * loss_ibs_angle
                    
        train_total_loss_dense += loss_dense.item()
        train_total_loss_coarse += loss_coarse.item()
        train_total_loss_medial_axis_surface += loss_medial_axis_surface.item()
        train_total_loss_medial_axis_interaction += loss_medial_axis_interaction.item()
        train_total_loss_ibs_angle += loss_ibs_angle.item()
        train_total_intersect_num += intersect_num.item()

        loss_total.backward()
        optimizer.step()

    for scheduler_item in lr_scheduler:
        scheduler_item.step()

    record_loss_info(specs, "train_loss_dense", train_total_loss_dense / train_dataloader.__len__(), epoch, tensorboard_writer)
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
        test_total_coarse = 0
        test_total_medial_axis_surface = 0
        test_total_medial_axis_interaction = 0
        test_total_ibs_angle = 0
        test_total_intersect_num = 0
        test_total_emd = 0
        for data, idx in test_dataloader:
            center, radius, direction, pcd_partial, pcd_gt = data
            pcd_partial = pcd_partial.to(device)

            coarse, dense = network(pcd_partial)

            pcd_gt = pcd_gt.to(device)
            center = center.to(device)
            radius = radius.to(device)
            direction = direction.to(device)

            loss_dense = cd_loss_L1(dense, pcd_gt)
            loss_coarse = cd_loss_L1(coarse, pcd_gt)
            loss_medial_axis_surface = medial_axis_surface_loss(center, radius, dense)
            loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, dense)
            loss_ibs_angle, intersect_num = ibs_angle_loss(center, radius, direction, dense)
            loss_emd = emd_loss(fps(dense, 2048), pcd_gt)

            test_total_dense += loss_dense.item()
            test_total_coarse += loss_coarse.item()
            test_total_medial_axis_surface += loss_medial_axis_surface.item()
            test_total_medial_axis_interaction += loss_medial_axis_interaction.item()
            test_total_ibs_angle += loss_ibs_angle.item()
            test_total_intersect_num += intersect_num.item()
            test_total_emd += loss_emd.item()

        test_avrg_dense = test_total_dense / test_dataloader.__len__()
        record_loss_info(specs, "test_loss_dense", test_total_dense / test_dataloader.__len__(), epoch, tensorboard_writer)
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
        save_model(specs, network, epoch)

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
    network = get_network(specs, PoinTr, checkpoint)
    optimizer = get_optimizer(network)
    lr_scheduler = get_lr_scheduler(network, optimizer)
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
        default="configs/INTE/train/specs_train_PoinTr_INTE.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
