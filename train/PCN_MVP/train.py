import sys

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")
import os.path

from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils
import torch.optim as Optim
import json
import open3d as o3d
import argparse
import time

from models.model_PCN_MVP import *

from utils import path_utils, log_utils
from utils.loss import cd_loss_L1, emd_loss
from dataset import dataset_MVP

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
    batch_size = specs.get("TrainOptions").get("BatchSize")
    num_data_loader_threads = specs.get("TrainOptions").get("DataLoaderThreads")

    logger.info("batch_size: {}".format(batch_size))
    logger.info("dataLoader threads: {}".format(num_data_loader_threads))

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    # get dataset
    train_dataset = dataset_MVP.PcdDataset(data_source, train_split)
    test_dataset = dataset_MVP.PcdDataset(data_source, test_split)

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

    network = PCN(num_dense=2048).to(device)

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


def train(network, train_dataloader, lr_schedule, optimizer, epoch, specs, tensorboard_writer):
    device = specs.get("Device")
    coarse_loss = specs.get("TrainOptions").get("CoarseLoss")

    network.train()
    logger.info("")
    logger.info('epoch: {}, learning rate: {}'.format(epoch, optimizer.param_groups[0]["lr"]))

    alpha = 0
    train_step = epoch * train_dataloader.__len__()
    if train_step < 10000:
        alpha = 0.01
    elif train_step < 20000:
        alpha = 0.1
    elif train_step < 50000:
        alpha = 0.5
    else:
        alpha = 1.0
    train_total_loss_coarse = 0
    train_total_loss_dense = 0
    train_total_loss = 0
    for pcd_partial, pcd_gt, idx in train_dataloader:
        optimizer.zero_grad()

        pcd_partial = pcd_partial.to(device)
        pcd_gt = pcd_gt.to(device)

        coarse_pred, dense_pred = network(pcd_partial)

        if coarse_loss == 'cd':
            loss_coarse = cd_loss_L1(coarse_pred, pcd_gt)
        elif coarse_loss == 'emd':
            coarse_gt = pcd_gt[:, :128, :]
            loss_coarse = emd_loss(coarse_pred, coarse_gt)
        else:
            raise ValueError('Not implemented loss {}'.format(coarse_loss))

        loss_dense = cd_loss_L1(dense_pred, pcd_gt)
        loss = loss_coarse + alpha * loss_dense

        train_total_loss_coarse += loss_coarse.item()
        train_total_loss_dense += loss_dense.item()
        train_total_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    lr_schedule.step()

    train_avrg_loss_coarse = train_total_loss_coarse / train_dataloader.__len__()
    tensorboard_writer.add_scalar("train_loss_coarse", train_avrg_loss_coarse, epoch)
    logger.info('train_avrg_loss_coarse: {}'.format(train_avrg_loss_coarse))

    train_avrg_loss_dense = train_total_loss_dense / train_dataloader.__len__()
    tensorboard_writer.add_scalar("train_loss_dense", train_avrg_loss_dense, epoch)
    logger.info('train_avrg_loss_dense: {}'.format(train_avrg_loss_dense))

    train_avrg_loss = train_total_loss / train_dataloader.__len__()
    tensorboard_writer.add_scalar("train_loss", train_avrg_loss, epoch)
    logger.info('train_avrg_loss: {}'.format(train_avrg_loss))


def test(network, test_dataloader, epoch, specs, tensorboard_writer, best_cd_l1, best_epoch):
    device = specs.get("Device")
    para_save_dir = specs.get("ParaSaveDir")

    network.eval()
    with torch.no_grad():
        test_total_cd_l1 = 0
        for pcd_partial, pcd_gt, idx in test_dataloader:
            pcd_partial = pcd_partial.to(device)
            pcd_gt = pcd_gt.to(device)

            coarse_pred, dense_pred = network(pcd_partial)
            loss_cd_l1 = cd_loss_L1(dense_pred, pcd_gt)

            test_total_cd_l1 += loss_cd_l1.item()

        test_avrg_loss_cd_l1 = test_total_cd_l1 / test_dataloader.__len__()
        tensorboard_writer.add_scalar("test_loss_cd_l1", test_avrg_loss_cd_l1, epoch)
        logger.info('test_avrg_loss_cd: {}'.format(test_avrg_loss_cd_l1))

        if test_avrg_loss_cd_l1 < best_cd_l1:
            best_epoch = epoch
            best_cd_l1 = test_avrg_loss_cd_l1
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
        default="configs/specs/specs_train_PCN_MVP.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = log_utils.get_train_logger(specs)
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
