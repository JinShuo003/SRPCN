import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path

from datetime import datetime, timedelta
import tensorboard.summary
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils
import torch.optim as Optim
import json
import argparse
import time

from models.PointAttN import PointAttN
from utils import path_utils, log_utils
from utils.loss import cd_loss_L1, medial_axis_surface_loss, medial_axis_interaction_loss
from dataset import data_INTE_norm

logger = None


def get_dataloader(specs):
    data_source = specs.get("DataSource")
    train_split_file = specs.get("TrainSplit")
    test_split_file = specs.get("TestSplit")
    batch_size = specs.get("TrainOptions").get("BatchSize")
    num_data_loader_threads = specs.get("TrainOptions").get("DataLoaderThreads")

    logger.info("batch_size: {}".format(batch_size))
    logger.info("dataLoader threads: {}".format(num_data_loader_threads))

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = data_INTE_norm.INTENormDataset(data_source, train_split)
    test_dataset = data_INTE_norm.INTENormDataset(data_source, test_split)

    logger.info("length of train_dataset: {}".format(train_dataset.__len__()))
    logger.info("length of test_dataset: {}".format(test_dataset.__len__()))

    # get dataloader
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logger.info("length of train_dataloader: {}".format(train_loader.__len__()))
    logger.info("length of test_dataloader: {}".format(test_loader.__len__()))

    return train_loader, test_loader


def get_network(specs):
    device = specs.get("Device")
    continue_train = specs.get("TrainOptions").get("ContinueTrain")

    network = PointAttN().to(device)

    if continue_train:
        continue_from_epoch = specs.get("TrainOptions").get("ContinueFromEpoch")
        para_save_dir = specs.get("ParaSaveDir")
        para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
        model_path = os.path.join(para_save_path, "epoch_{}.pth".format(continue_from_epoch))
        logger.info("load model from {}".format(model_path))
        state_dict = torch.load(model_path, map_location="cuda:{}".format(device))
        network.load_state_dict(state_dict)

    if torch.cuda.is_available():
        network = network.to(device)
    return network


def get_optimizer(specs, network):
    learning_rate = specs.get("TrainOptions").get("LearningRate")
    optimizer = Optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    return lr_schedual, optimizer


def get_tensorboard_writer(specs, network):
    device = specs.get("Device")

    writer_path = os.path.join(specs.get("TensorboardLogDir"), specs.get("TAG"))
    if not os.path.isdir(writer_path):
        os.makedirs(writer_path)

    tensorboard_writer = SummaryWriter(writer_path)

    input_pcd_shape = torch.randn(1, specs.get("PcdPointNum"), 3)

    if torch.cuda.is_available():
        input_pcd_shape = input_pcd_shape.to(device)

    tensorboard_writer.add_graph(network, input_pcd_shape)

    return tensorboard_writer


def save_model(specs, model, epoch):
    para_save_dir = specs.get("ParaSaveDir")
    para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
    if not os.path.isdir(para_save_path):
        os.mkdir(para_save_path)
    model_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_filename)


def get_medial_axis_loss_weight(specs, epoch):
    begin_epoch = specs.get("MedialAxisLossOptions").get("BeginEpoch")
    init_ratio = specs.get("MedialAxisLossOptions").get("InitRatio")
    step_size = specs.get("MedialAxisLossOptions").get("StepSize")
    gamma = specs.get("MedialAxisLossOptions").get("Gamma")
    return init_ratio * pow((1 + gamma), (epoch - begin_epoch) / step_size)


def record_loss_info(tag: str, avrg_loss, epoch, tensorboard_writer: SummaryWriter):
    tensorboard_writer.add_scalar("{}".format(tag), avrg_loss, epoch)
    logger.info('{}: {}'.format(tag, avrg_loss))


def train(network, train_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer):
    device = specs.get("Device")

    network.train()
    logger.info("")
    logger.info('epoch: {}, learning rate: {}'.format(epoch, optimizer.param_groups[0]["lr"]))

    medial_axis_loss_weight = get_medial_axis_loss_weight(specs, epoch)
    logger.info("medial_axis_loss_weight: {}".format(medial_axis_loss_weight))

    train_total_loss_dense = 0
    train_total_loss_sub_dense = 0
    train_total_loss_coarse = 0
    train_total_loss_medial_axis_surface = 0
    for center, radius, pcd_partial, pcd_gt, idx in train_dataloader:
        optimizer.zero_grad()

        pcd_partial = pcd_partial.to(device)
        pcd_pred_coarse, pcd_pred_sub_dense, pcd_pred_dense = network(pcd_partial)

        pcd_gt = pcd_gt.to(device)

        loss_dense = cd_loss_L1(pcd_pred_dense, pcd_gt)
        gt_sub_dense = gather_operation(pcd_gt.transpose(1, 2).contiguous(),
                                        furthest_point_sample(pcd_gt, pcd_pred_sub_dense.shape[1])).transpose(1,
                                                                                                              2).contiguous()
        loss_sub_dense = cd_loss_L1(pcd_pred_sub_dense, gt_sub_dense)
        gt_coarse = gather_operation(gt_sub_dense.transpose(1, 2).contiguous(),
                                     furthest_point_sample(gt_sub_dense, pcd_pred_coarse.shape[1])).transpose(1,
                                                                                                              2).contiguous()
        loss_coarse = cd_loss_L1(pcd_pred_dense, gt_coarse)

        loss_medial_axis_surface = medial_axis_surface_loss(center, radius, pcd_pred_dense)

        loss_total = loss_dense + loss_sub_dense + loss_coarse + medial_axis_loss_weight * loss_medial_axis_surface

        train_total_loss_dense += loss_dense.item()
        train_total_loss_sub_dense += loss_sub_dense.item()
        train_total_loss_coarse += loss_coarse.item()
        train_total_loss_medial_axis_surface += loss_medial_axis_surface.item()

        loss_total.backward()
        optimizer.step()

    lr_schedule.step()

    record_loss_info("train_loss_dense", train_total_loss_dense / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info("train_loss_sub_dense", train_total_loss_sub_dense / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info("train_loss_coarse", train_total_loss_coarse / train_dataloader.__len__(), epoch,
                     tensorboard_writer)
    record_loss_info("train_loss_medial_axis_surface",
                     train_total_loss_medial_axis_surface / train_dataloader.__len__(), epoch, tensorboard_writer)


def test(network, test_dataloader, epoch, specs, tensorboard_writer, best_cd_l1, best_epoch):
    device = specs.get("Device")

    network.eval()
    with torch.no_grad():
        test_total_dense = 0
        test_total_sub_dense = 0
        test_total_coarse = 0
        test_total_medial_axis_surface = 0
        test_total_medial_axis_interaction = 0
        for center, radius, pcd_partial, pcd_gt, idx in test_dataloader:
            pcd_partial = pcd_partial.to(device)

            pcd_pred_coarse, pcd_pred_sub_dense, pcd_pred_dense = network(pcd_partial)

            pcd_gt = pcd_gt.to(device)

            loss_dense = cd_loss_L1(pcd_pred_dense, pcd_gt)
            loss_sub_dense = cd_loss_L1(pcd_pred_sub_dense, pcd_gt)
            loss_coarse = cd_loss_L1(pcd_pred_coarse, pcd_gt)
            loss_medial_axis_surface = medial_axis_surface_loss(center, radius, pcd_pred_dense)
            loss_medial_axis_interaction = medial_axis_interaction_loss(center, radius, pcd_pred_dense)

            test_total_dense += loss_dense.item()
            test_total_sub_dense += loss_sub_dense.item()
            test_total_coarse += loss_coarse.item()
            test_total_medial_axis_surface += loss_medial_axis_surface.item()
            test_total_medial_axis_interaction += loss_medial_axis_interaction.item()

        test_avrg_dense = test_total_dense / test_dataloader.__len__()
        record_loss_info("test_loss_dense", test_total_dense / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info("test_sub_dense", test_total_sub_dense / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info("test_loss_coarse", test_total_coarse / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info("test_loss_medial_axis_surface", test_total_medial_axis_surface / test_dataloader.__len__(),
                         epoch, tensorboard_writer)
        record_loss_info("test_loss_medial_axis_interaction",
                         test_total_medial_axis_interaction / test_dataloader.__len__(), epoch, tensorboard_writer)

        if test_avrg_dense / test_dataloader.__len__() < best_cd_l1:
            best_epoch = epoch
            best_cd_l1 = test_avrg_dense / test_dataloader.__len__()
            logger.info('newest best epoch: {}'.format(best_epoch))
            logger.info('newest best cd l1: {}'.format(best_cd_l1))
            save_model(specs, network, epoch)
        if epoch % 5 == 0:
            save_model(specs, network, epoch)

        return best_cd_l1, best_epoch


def main_function(specs):
    epoch_num = specs.get("TrainOptions").get("NumEpochs")
    continue_train = specs.get("TrainOptions").get("ContinueTrain")
    continue_from_epoch = specs.get("TrainOptions").get("ContinueFromEpoch")

    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))

    logger.info("current network TAG: {}".format(specs.get("TAG")))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("There are {} epochs in total".format(epoch_num))

    train_loader, test_loader = get_dataloader(specs)
    network = get_network(specs)
    lr_schedule, optimizer = get_optimizer(specs, network)
    tensorboard_writer = get_tensorboard_writer(specs, network)

    best_cd_l1 = 1e8
    best_epoch = -1
    epoch_begin = 0
    if continue_train:
        epoch_begin = continue_from_epoch + 1
        logger.info("continue train from epoch {}".format(epoch_begin))
    for epoch in range(epoch_begin, epoch_num + 1):
        time_begin_train = time.time()
        train(network, train_loader, lr_schedule, optimizer, epoch, specs, tensorboard_writer)
        time_end_train = time.time()
        logger.info("use {} to train".format(time_end_train - time_begin_train))

        time_begin_test = time.time()
        best_cd_l1, best_epoch = test(network, test_loader, epoch, specs, tensorboard_writer, best_cd_l1, best_epoch)
        time_end_test = time.time()
        logger.info("use {} to test".format(time_end_test - time_begin_test))

    tensorboard_writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train IBPCDC")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_train_PointAttN_INTE.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = log_utils.get_train_logger(specs)
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
