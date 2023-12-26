import sys
sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import logging
import os.path

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import json
import open3d as o3d
from datetime import datetime, timedelta

import networks.loss
from networks.model_transformer_TopNet_2obj_no_ibs import *

import utils.data
import utils.workspace as ws
from utils.learning_rate import get_learning_rate_schedules


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
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]
    scene_per_batch = specs["ScenesPerBatch"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    logging.info("batch_size: {}".format(scene_per_batch))
    logging.info("dataLoader threads: {}".format(num_data_loader_threads))

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = utils.data.IntersectDataset(data_source, train_split)
    test_dataset = utils.data.IntersectDataset(data_source, test_split)

    logging.info("length of sdf_train_dataset: {}".format(train_dataset.__len__()))
    logging.info("length of sdf_test_dataset: {}".format(test_dataset.__len__()))

    # get dataloader
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    logging.info("length of sdf_train_loader: {}".format(train_loader.__len__()))
    logging.info("length of sdf_test_loader: {}".format(test_loader.__len__()))

    return train_loader, test_loader


def get_network(specs):
    device = specs["Device"]

    net = IBPCDCNet()

    if torch.cuda.is_available():
        net = net.to(device)
    return net


def get_optimizer(specs, IBS_Net):
    lr_schedules = get_learning_rate_schedules(specs)
    optimizer = torch.optim.Adam(IBS_Net.parameters(), lr_schedules.get_learning_rate(0))

    return lr_schedules, optimizer


def get_tensorboard_writer(specs, log_path, network, TIMESTAMP):
    device = specs["Device"]
    train_split_file = specs["TrainSplit"]

    writer_path = os.path.join(log_path, "{}_{}".format(os.path.basename(train_split_file).split('.')[-2], TIMESTAMP))
    if os.path.isdir(writer_path):
        os.mkdir(writer_path)

    tensorboard_writer = SummaryWriter(writer_path)

    input_pcd1_shape = torch.randn(1, specs.get("PcdPointNum"), 3)
    input_pcd2_shape = torch.randn(1, specs.get("PcdPointNum"), 3)

    if torch.cuda.is_available():
        input_pcd1_shape = input_pcd1_shape.to(device)
        input_pcd2_shape = input_pcd2_shape.to(device)

    tensorboard_writer.add_graph(network, (input_pcd1_shape, input_pcd2_shape))

    return tensorboard_writer


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def train(network, sdf_train_loader, lr_schedules, optimizer, epoch, specs, tensorboard_writer, TIMESTAMP):
    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        optimizer.param_groups[0]["lr"] = lr_schedules.get_learning_rate(epoch)

    para_save_dir = specs["ParaSaveDir"]
    train_split_file = specs["TrainSplit"]
    device = specs["Device"]

    loss_emd = networks.loss.emdModule()
    loss_cd = networks.loss.cdModule()

    network.train()
    adjust_learning_rate(lr_schedules, optimizer, epoch)
    logging.info("")
    logging.info('epoch: {}, learning rate: {}'.format(epoch, lr_schedules.get_learning_rate(epoch)))

    train_total_loss_emd = 0
    train_total_loss_cd = 0
    for IBS, pcd1_partial, pcd2_partial, pcd1_gt, pcd2_gt, idx in sdf_train_loader:
        pcd1_partial.requires_grad = False
        pcd2_partial.requires_grad = False
        pcd1_gt.requires_grad = False
        pcd2_gt.requires_grad = False

        pcd1_partial = pcd1_partial.to(device)
        pcd2_partial = pcd2_partial.to(device)
        pcd1_out, pcd2_out = network(pcd1_partial, pcd2_partial)

        pcd1_gt = pcd1_gt.to(device)
        pcd2_gt = pcd2_gt.to(device)
        loss_emd_pcd1 = torch.mean(loss_emd(pcd1_out, pcd1_gt)[0])
        loss_emd_pcd2 = torch.mean(loss_emd(pcd2_out, pcd2_gt)[0])
        loss_cd_pcd1 = loss_cd(pcd1_out, pcd1_gt)
        loss_cd_pcd2 = loss_cd(pcd2_out, pcd2_gt)

        batch_loss_emd = loss_emd_pcd1 + loss_emd_pcd2
        batch_loss_cd = loss_cd_pcd1 + loss_cd_pcd2

        train_total_loss_emd += batch_loss_emd.item()
        train_total_loss_cd += batch_loss_cd.item()

        optimizer.zero_grad()
        batch_loss_emd.backward()
        optimizer.step()

    train_avrg_loss_emd = train_total_loss_emd / sdf_train_loader.__len__()
    tensorboard_writer.add_scalar("train_loss_emd", train_avrg_loss_emd, epoch)
    logging.info('train_avrg_loss_emd: {}'.format(train_avrg_loss_emd))
    train_avrg_loss_cd = train_total_loss_cd / sdf_train_loader.__len__()
    tensorboard_writer.add_scalar("train_loss_cd", train_avrg_loss_cd, epoch)
    logging.info('train_avrg_loss_emd: {}'.format(train_avrg_loss_cd))

    # 保存模型
    if epoch % 5 == 0:
        para_save_path = os.path.join(para_save_dir,
                                      "{}_{}".format(os.path.basename(train_split_file).split('.')[-2], TIMESTAMP))
        if not os.path.isdir(para_save_path):
            os.mkdir(para_save_path)
        model_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))
        torch.save(network, model_filename)


def test(network, test_dataloader, epoch, specs, tensorboard_writer):
    device = specs["Device"]

    loss_emd = networks.loss.emdModule()
    loss_cd = networks.loss.cdModule()

    with torch.no_grad():
        test_total_loss_emd = 0
        test_total_loss_cd = 0
        for IBS, pcd1_partial, pcd2_partial, pcd1_gt, pcd2_gt, idx in test_dataloader:
            pcd1_partial.requires_grad = False
            pcd2_partial.requires_grad = False
            pcd1_gt.requires_grad = False
            pcd2_gt.requires_grad = False

            IBS = IBS.to(device)
            pcd1_partial = pcd1_partial.to(device)
            pcd2_partial = pcd2_partial.to(device)
            pcd1_out, pcd2_out = network(pcd1_partial, pcd2_partial)

            pcd1_gt = pcd1_gt.to(device)
            pcd2_gt = pcd2_gt.to(device)
            loss_emd_pcd1 = torch.mean(loss_emd(pcd1_out, pcd1_gt)[0])
            loss_emd_pcd2 = torch.mean(loss_emd(pcd2_out, pcd2_gt)[0])
            loss_cd_pcd1 = loss_cd(pcd1_out, pcd1_gt)
            loss_cd_pcd2 = loss_cd(pcd2_out, pcd2_gt)

            batch_loss_emd = (loss_emd_pcd1 + loss_emd_pcd2)
            batch_loss_cd = loss_cd_pcd1 + loss_cd_pcd2

            test_total_loss_emd += batch_loss_emd.item()
            test_total_loss_cd += batch_loss_cd.item()

        test_avrg_loss_emd = test_total_loss_emd / test_dataloader.__len__()
        tensorboard_writer.add_scalar("test_loss_emd", test_avrg_loss_emd, epoch)
        logging.info('test_avrg_loss_emd: {}'.format(test_avrg_loss_emd))
        test_avrg_loss_cd = test_total_loss_cd / test_dataloader.__len__()
        tensorboard_writer.add_scalar("test_loss_cd", test_avrg_loss_cd, epoch)
        logging.info('test_avrg_loss_cd: {}'.format(test_avrg_loss_cd))


def main_function(experiment_config_file):
    specs = ws.load_experiment_specifications(experiment_config_file)
    epoch_num = specs["NumEpochs"]
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))

    logging.info("current experiment config file: {}".format(experiment_config_file))
    logging.info("current time: {}".format(TIMESTAMP))
    logging.info("There are {} epochs in total".format(epoch_num))

    train_loader, test_loader = get_dataloader(specs)
    IBS_Net = get_network(specs)
    lr_schedules, optimizer = get_optimizer(specs, IBS_Net)
    tensorboard_writer = get_tensorboard_writer(specs, './tensorboard_logs', IBS_Net, TIMESTAMP)

    for epoch in range(epoch_num):
        train(IBS_Net, train_loader, lr_schedules, optimizer, epoch, specs, tensorboard_writer, TIMESTAMP)
        test(IBS_Net, test_loader, epoch, specs, tensorboard_writer)

    tensorboard_writer.close()


if __name__ == '__main__':
    import argparse
    from utils.cmd_utils import add_common_args
    from utils.logging import configure_logging

    arg_parser = argparse.ArgumentParser(description="Train a IBS Net")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/specs/specs_train.json",
        required=False,
        help="The experiment config file."
    )

    # 添加日志参数
    add_common_args(arg_parser)

    args = arg_parser.parse_args()

    configure_logging(args)

    main_function(args.experiment_config_file)
