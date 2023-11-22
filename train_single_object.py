import logging
import os.path

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import json
import open3d as o3d
from datetime import datetime, timedelta

import networks.loss
from networks.models import *

import utils.data
import utils.workspace as ws
from utils.learning_rate import get_learning_rate_schedules


def visualize_data1(pcd1, pcd2, IBS, pcd1_gt, pcd2_gt):
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
    dataloader_cache_capacity = specs["DataLoaderSpecs"]["CacheCapacity"]
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    logging.info("batch_size: {}".format(scene_per_batch))
    logging.info("dataLoader threads: {}".format(num_data_loader_threads))

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = utils.data.InterceptDataset(data_source, train_split, dataloader_cache_capacity)
    test_dataset = utils.data.InterceptDataset(data_source, test_split, dataloader_cache_capacity)

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
    pcd_point_num = specs["PcdPointNum"]
    device = specs["Device"]

    net = PCDCompletionNet(pcd_point_num)

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

    input_pcd1_shape = torch.randn(1, 512, 3)
    input_pcd2_shape = torch.randn(1, 512, 3)
    input_IBS_shape = torch.randn(1, 512, 3)

    if torch.cuda.is_available():
        input_pcd1_shape = input_pcd1_shape.to(device)
        input_pcd2_shape = input_pcd2_shape.to(device)
        input_IBS_shape = input_IBS_shape.to(device)

    tensorboard_writer.add_graph(network, (input_pcd1_shape, input_pcd2_shape, input_IBS_shape))

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
    loss_weight_cd = specs["LossSpecs"]["WeightCD"]
    loss_weight_emd = specs["LossSpecs"]["WeightEMD"]

    loss_cd = networks.loss.cdModule()
    loss_emd = networks.loss.emdModule()

    network.train()
    adjust_learning_rate(lr_schedules, optimizer, epoch)
    logging.info('epoch: {}, learning rate: {}'.format(epoch, lr_schedules.get_learning_rate(epoch)))

    train_total_loss = 0
    for IBS, pcd1, pcd2, pcd1gt, pcd2gt, idx in sdf_train_loader:
        pcd1 = pcd1.to(device)
        pcd2 = pcd2.to(device)
        IBS = IBS.to(device)

        # visualize_data1(pcd1, pcd2, IBS, pcd1gt, pcd2gt)
        pcd1_out = network(pcd1, IBS)
        pcd2_out = network(pcd2, IBS)

        # loss between out and groundtruth
        loss_cd_pcd1 = loss_cd(pcd1, pcd1_out)
        loss_cd_pcd2 = loss_cd(pcd2, pcd2_out)
        loss_emd_pcd1 = loss_emd(pcd1, pcd1_out)
        loss_emd_pcd2 = loss_emd(pcd2, pcd2_out)

        batch_loss = loss_weight_cd * (loss_cd_pcd1 + loss_cd_pcd2) + loss_weight_emd * (loss_emd_pcd1 + loss_emd_pcd2)

        # 统计一个epoch的平均loss
        train_total_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    train_avrg_loss = train_total_loss / sdf_train_loader.__len__()
    tensorboard_writer.add_scalar("train_loss", train_avrg_loss, epoch)
    logging.info('train_avrg_loss: {}'.format(train_avrg_loss))

    # 保存模型
    if epoch % 5 == 0:
        para_save_path = os.path.join(para_save_dir, "{}_{}".format(os.path.basename(train_split_file).split('.')[-2], TIMESTAMP))
        if not os.path.isdir(para_save_path):
            os.mkdir(para_save_path)
        model_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))
        torch.save(network, model_filename)


def test(network, test_dataloader, epoch, specs, tensorboard_writer):
    device = specs["Device"]
    loss_weight_cd = specs["LossSpecs"]["WeightCD"]
    loss_weight_emd = specs["LossSpecs"]["WeightEMD"]

    loss_cd = networks.loss.cdModule()
    loss_emd = networks.loss.emdModule()

    with torch.no_grad():
        test_total_loss = 0
        for IBS, pcd1, pcd2, pcd1gt, pcd2gt, idx in test_dataloader:
            pcd1 = pcd1.to(device)
            pcd2 = pcd2.to(device)
            IBS = IBS.to(device)

            # visualize_data1(pcd1, pcd2, xyz, udf_gt1, udf_gt2)
            pcd1_out = network(pcd1, IBS)
            pcd2_out = network(pcd2, IBS)

            loss_cd_pcd1 = loss_cd(pcd1, pcd1_out)
            loss_cd_pcd2 = loss_cd(pcd2, pcd2_out)
            loss_emd_pcd1 = loss_emd(pcd1, pcd1_out)
            loss_emd_pcd2 = loss_emd(pcd2, pcd2_out)

            batch_loss = loss_weight_cd * (loss_cd_pcd1 + loss_cd_pcd2) + loss_weight_emd * (
                        loss_emd_pcd1 + loss_emd_pcd2)

            test_total_loss += batch_loss.item()

        test_avrg_loss = test_total_loss / test_dataloader.__len__()
        tensorboard_writer.add_scalar("test_loss", test_avrg_loss, epoch)
        logging.info(' test_avrg_loss: {}\n'.format(test_avrg_loss))


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
    from utils.cmd import add_common_args
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
