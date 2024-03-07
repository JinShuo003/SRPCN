import sys
import os

sys.path.insert(0, os.path.abspath("."))

import os.path

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from datetime import datetime, timedelta
import argparse
import time
import torch.nn as nn
from torch import optim

from models.AdaPoinTr import AdaPoinTr
from models.pn2_utils import fps_subsample
from utils import path_utils
from utils.loss import cd_loss_L1, cd_loss_L1_single, emd_loss
from utils.train_utils import *
from dataset import dataset_C3d


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
        for name, param in model.module.named_parameters():
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


def train(network, train_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")

    network.train()
    logger.info("")
    logger.info('epoch: {}, learning rate: {}'.format(epoch, optimizer.param_groups[0]["lr"]))

    train_total_loss_dense = 0
    train_total_loss_denoised = 0
    train_total_loss_coarse = 0

    for data, idx in train_dataloader:
        pcd_partial, pcd_gt = data
        optimizer.zero_grad()

        pcd_partial = pcd_partial.to(device)
        ret = network(pcd_partial)

        pcd_gt = pcd_gt.to(device)

        loss_denoised, loss_coarse, loss_fine = network.module.get_loss(ret, pcd_gt, epoch)

        loss_total = loss_denoised + loss_coarse + loss_fine

        train_total_loss_dense += loss_fine.item()
        train_total_loss_denoised += loss_denoised.item()
        train_total_loss_coarse += loss_coarse.item()

        loss_total.backward()
        optimizer.step()

    lr_schedule.step()

    record_loss_info(specs, "train_loss_dense", train_total_loss_dense / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info(specs, "train_loss_sub_dense", train_total_loss_denoised / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info(specs, "train_loss_coarse", train_total_loss_coarse / train_dataloader.__len__(), epoch,
                     tensorboard_writer)


def test(network, test_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer, best_cd, best_epoch):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")

    network.eval()
    with torch.no_grad():
        test_total_dense = 0
        test_total_denoise = 0
        test_total_coarse = 0
        test_total_emd = 0
        for data, idx in test_dataloader:
            pcd_partial, pcd_gt = data
            pcd_partial = pcd_partial.to(device)

            ret = network(pcd_partial)
            pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

            pcd_gt = pcd_gt.to(device)

            loss_dense = cd_loss_L1(pred_fine, pcd_gt)
            loss_coarse = cd_loss_L1(pred_coarse, pcd_gt)
            loss_denoise = cd_loss_L1(denoised_fine, pcd_gt)

            loss_emd = emd_loss(pred_fine, pcd_gt)

            test_total_dense += loss_dense.item()
            test_total_denoise += loss_denoise.item()
            test_total_coarse += loss_coarse.item()
            test_total_emd += loss_emd.item()

        test_avrg_dense = test_total_dense / test_dataloader.__len__()
        record_loss_info(specs, "test_loss_dense", test_total_dense / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
        record_loss_info(specs, "test_loss_sub_dense", test_total_denoise / test_dataloader.__len__(), epoch,
                         tensorboard_writer)
        record_loss_info(specs, "test_loss_coarse", test_total_coarse / test_dataloader.__len__(), epoch,
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

    train_loader, test_loader = get_dataloader(dataset_C3d.C3dDataset, specs)
    checkpoint = None
    network = get_network(specs, AdaPoinTr, checkpoint)
    optimizer = get_optimizer(network)
    lr_scheduler = get_lr_scheduler(optimizer)
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
        default="configs/C3d/train/specs_train_AdaPointTr_C3d.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
