import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import json
import open3d as o3d
from datetime import datetime, timedelta
import argparse
import time

from networks.loss import chamfer_distance, earth_move_distance
from networks.model_Transformer_TopNet_2obj_ibs import *

import utils.data
from utils.learning_rate import get_learning_rate_schedules
from utils import path_utils, log_utils

logger = None


def visualize(pcd1, pcd2, IBS, pcd1_gt, pcd2_gt):
    # 将udf数据拆分开，并且转移到cpu
    IBS = IBS.cpu().detach().numpy()
    pcd1_np = pcd1.cpu().detach().numpy()
    pcd2_np = pcd2.cpu().detach().numpy()
    pcd1gt_np = pcd1_gt.cpu().detach().numpy()
    pcd2gt_np = pcd2_gt.cpu().detach().numpy()

    for i in range(pcd1_np.shape[0]):
        pcd1_o3d = o3d.geometry.PointCloud()
        pcd2_o3d = o3d.geometry.PointCloud()
        ibs_o3d = o3d.geometry.PointCloud()
        pcd1gt_o3d = o3d.geometry.PointCloud()
        pcd2gt_o3d = o3d.geometry.PointCloud()

        pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1_np[i])
        pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2_np[i])
        ibs_o3d.points = o3d.utility.Vector3dVector(IBS[i])
        pcd1gt_o3d.points = o3d.utility.Vector3dVector(pcd1gt_np[i])
        pcd2gt_o3d.points = o3d.utility.Vector3dVector(pcd2gt_np[i])

        pcd1_o3d.paint_uniform_color([1, 0, 0])
        pcd2_o3d.paint_uniform_color([0, 1, 0])
        ibs_o3d.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([ibs_o3d, pcd1_o3d, pcd2_o3d])


def get_dataloader(specs):
    data_source = specs.get("DataSource")
    train_split_file = specs.get("TrainSplit")
    test_split_file = specs.get("TestSplit")
    batch_size = specs.get("BatchSize")
    num_data_loader_threads = specs.get("DataLoaderThreads")

    logger.info("batch_size: {}".format(batch_size))
    logger.info("dataLoader threads: {}".format(num_data_loader_threads))

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = utils.data.IntersectDataset(data_source, train_split)
    test_dataset = utils.data.IntersectDataset(data_source, test_split)

    logger.info("length of sdf_train_dataset: {}".format(train_dataset.__len__()))
    logger.info("length of sdf_test_dataset: {}".format(test_dataset.__len__()))

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
    logger.info("length of sdf_train_loader: {}".format(train_loader.__len__()))
    logger.info("length of sdf_test_loader: {}".format(test_loader.__len__()))

    return train_loader, test_loader


def get_network(specs):
    device = specs.get("Device")

    net = IBPCDCNet()

    if torch.cuda.is_available():
        net = net.to(device)
    return net


def get_optimizer(specs, network):
    lr_schedules = get_learning_rate_schedules(specs)
    optimizer = torch.optim.Adam(network.parameters(), lr_schedules.get_learning_rate(0))

    return lr_schedules, optimizer


def get_tensorboard_writer(specs, network):
    device = specs.get("Device")

    writer_path = os.path.join(specs.get("TensorboardLogDir"), specs.get("TAG"))
    if not os.path.isdir(writer_path):
        os.makedirs(writer_path)

    tensorboard_writer = SummaryWriter(writer_path)

    input_pcd1_shape = torch.randn(1, specs.get("PcdPointNum"), 3)
    input_pcd2_shape = torch.randn(1, specs.get("PcdPointNum"), 3)
    input_IBS_shape = torch.randn(1, specs.get("IBSPointNum"), 3)

    if torch.cuda.is_available():
        input_pcd1_shape = input_pcd1_shape.to(device)
        input_pcd2_shape = input_pcd2_shape.to(device)
        input_IBS_shape = input_IBS_shape.to(device)

    # tensorboard_writer.add_graph(network, (input_pcd1_shape, input_pcd2_shape, input_IBS_shape))

    return tensorboard_writer


def train(network, sdf_train_loader, lr_schedules, optimizer, epoch, specs, tensorboard_writer, TIMESTAMP):
    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        optimizer.param_groups[0]["lr"] = lr_schedules.get_learning_rate(epoch)

    para_save_dir = specs.get("ParaSaveDir")
    device = specs.get("Device")

    network.train()
    adjust_learning_rate(lr_schedules, optimizer, epoch)
    logger.info("")
    logger.info('epoch: {}, learning rate: {}'.format(epoch, lr_schedules.get_learning_rate(epoch)))

    train_total_loss_emd = 0
    train_total_loss_cd = 0
    for IBS, pcd1_partial, pcd2_partial, pcd1_gt, pcd2_gt, idx in sdf_train_loader:
        pcd1_partial = pcd1_partial.to(device)
        pcd2_partial = pcd2_partial.to(device)
        IBS = IBS.to(device)
        pcd1_out, pcd2_out = network(pcd1_partial, pcd2_partial, IBS)

        pcd1_gt = pcd1_gt.to(device)
        pcd2_gt = pcd2_gt.to(device)
        loss_emd_pcd1 = earth_move_distance(pcd1_out, pcd1_gt)
        loss_emd_pcd2 = earth_move_distance(pcd2_out, pcd2_gt)
        loss_cd_pcd1 = chamfer_distance(pcd1_out, pcd1_gt)
        loss_cd_pcd2 = chamfer_distance(pcd2_out, pcd2_gt)

        batch_loss_emd = loss_emd_pcd1 + loss_emd_pcd2
        batch_loss_cd = loss_cd_pcd1 + loss_cd_pcd2

        train_total_loss_emd += batch_loss_emd.item()
        train_total_loss_cd += batch_loss_cd.item()

        optimizer.zero_grad()
        batch_loss_emd.backward()
        optimizer.step()

    train_avrg_loss_emd = train_total_loss_emd / sdf_train_loader.__len__()
    tensorboard_writer.add_scalar("train_loss_emd", train_avrg_loss_emd, epoch)
    logger.info('train_avrg_loss_emd: {}'.format(train_avrg_loss_emd))

    train_avrg_loss_cd = train_total_loss_cd / sdf_train_loader.__len__()
    tensorboard_writer.add_scalar("train_loss_cd", train_avrg_loss_cd, epoch)
    logger.info('train_avrg_loss_cd: {}'.format(train_avrg_loss_cd))

    # 保存模型
    if epoch % 5 == 0:
        para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
        if not os.path.isdir(para_save_path):
            os.mkdir(para_save_path)
        model_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))
        torch.save(network, model_filename)


def test(network, test_dataloader, epoch, specs, tensorboard_writer):
    device = specs.get("Device")

    with torch.no_grad():
        test_total_loss_emd = 0
        test_total_loss_cd = 0
        for IBS, pcd1_partial, pcd2_partial, pcd1_gt, pcd2_gt, idx in test_dataloader:
            pcd1_partial = pcd1_partial.to(device)
            pcd2_partial = pcd2_partial.to(device)
            IBS = IBS.to(device)
            pcd1_out, pcd2_out = network(pcd1_partial, pcd2_partial, IBS)

            pcd1_gt = pcd1_gt.to(device)
            pcd2_gt = pcd2_gt.to(device)
            loss_emd_pcd1 = earth_move_distance(pcd1_out, pcd1_gt)
            loss_emd_pcd2 = earth_move_distance(pcd2_out, pcd2_gt)
            loss_cd_pcd1 = chamfer_distance(pcd1_out, pcd1_gt)
            loss_cd_pcd2 = chamfer_distance(pcd2_out, pcd2_gt)

            batch_loss_emd = (loss_emd_pcd1 + loss_emd_pcd2)
            batch_loss_cd = loss_cd_pcd1 + loss_cd_pcd2

            test_total_loss_emd += batch_loss_emd.item()
            test_total_loss_cd += batch_loss_cd.item()

        test_avrg_loss_emd = test_total_loss_emd / test_dataloader.__len__()
        tensorboard_writer.add_scalar("test_loss_emd", test_avrg_loss_emd, epoch)
        logger.info('test_avrg_loss_emd: {}'.format(test_avrg_loss_emd))

        test_avrg_loss_cd = test_total_loss_cd / test_dataloader.__len__()
        tensorboard_writer.add_scalar("test_loss_cd", test_avrg_loss_cd, epoch)
        logger.info('test_avrg_loss_cd: {}'.format(test_avrg_loss_cd))


def main_function(specs):
    epoch_num = specs["NumEpochs"]
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))

    logger.info("current network TAG: {}".format(specs.get("TAG")))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("There are {} epochs in total".format(epoch_num))

    train_loader, test_loader = get_dataloader(specs)
    network = get_network(specs)
    lr_schedules, optimizer = get_optimizer(specs, network)
    tensorboard_writer = get_tensorboard_writer(specs, network)

    for epoch in range(epoch_num + 1):
        time_begin_train = time.time()
        train(network, train_loader, lr_schedules, optimizer, epoch, specs, tensorboard_writer, TIMESTAMP)
        time_end_train = time.time()
        logger.info("use {} to train".format(time_end_train - time_begin_train))

        time_begin_test = time.time()
        test(network, test_loader, epoch, specs, tensorboard_writer)
        time_end_test = time.time()
        logger.info("use {} to test".format(time_end_test - time_begin_test))

    tensorboard_writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train IBPCDC")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_train_Transformer_TopNet_2obj_ibs.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = log_utils.get_train_logger(specs)

    logger.info("specs file path: {}".format(args.experiment_config_file))

    main_function(specs)
