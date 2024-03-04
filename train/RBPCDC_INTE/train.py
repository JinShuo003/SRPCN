import sys
import os

sys.path.insert(0, os.path.abspath("."))

import os.path
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from datetime import datetime, timedelta
import argparse
import time

from models.RBPCDC import TopNet_path1, TopNet_path2
from utils import path_utils, geometry_utils
from utils.loss import cd_loss_L1, medial_axis_surface_loss, medial_axis_interaction_loss, ibs_angle_loss, emd_loss
from utils.train_utils import *
from dataset import dataset_RBPCDC


def save_model(specs, model, epoch):
    para_save_dir = specs.get("ParaSaveDir")
    para_save_path = os.path.join(para_save_dir, specs.get("TAG"))
    if not os.path.isdir(para_save_path):
        os.mkdir(para_save_path)

    network_path1, network_path2 = model
    checkpoint = {
        "epoch": epoch,
        "model_path1": network_path1.state_dict(),
        "model_path2": network_path2.state_dict()
    }
    checkpoint_filename = os.path.join(para_save_path, "epoch_{}.pth".format(epoch))

    torch.save(checkpoint, checkpoint_filename)


def get_evaluation_metrics(pcd_pred, pcd_gt, pcd_normalize_para, medial_axis_sphere):
    pcd_pred_path1, pcd_pred_path2 = pcd_pred
    pcd1_pred_path1, pcd2_pred_path1 = pcd_pred_path1
    pcd1_pred_path2, pcd2_pred_path2 = pcd_pred_path2
    pcd1_gt, pcd2_gt = pcd_gt

    pcd1_normalize_para, pcd2_normalize_para = pcd_normalize_para
    pcd1_centroid, pcd1_scale = pcd1_normalize_para
    pcd2_centroid, pcd2_scale = pcd2_normalize_para

    medial_axis_sphere1, medial_axis_sphere2 = medial_axis_sphere
    center1, radius1, direction1,  = medial_axis_sphere1
    center2, radius2, direction2,  = medial_axis_sphere2

    pcd1_gt_origin = geometry_utils.denormalize_geometry_tensor_batch(pcd1_gt, pcd1_centroid, pcd1_scale)
    pcd1_pred_path1_origin = geometry_utils.denormalize_geometry_tensor_batch(pcd1_pred_path1, pcd1_centroid,
                                                                              pcd1_scale)
    pcd1_pred_path2_origin = geometry_utils.denormalize_geometry_tensor_batch(pcd1_pred_path2, pcd1_centroid,
                                                                              pcd1_scale)
    pcd2_gt_origin = geometry_utils.denormalize_geometry_tensor_batch(pcd2_gt, pcd2_centroid, pcd2_scale)
    pcd2_pred_path1_origin = geometry_utils.denormalize_geometry_tensor_batch(pcd2_pred_path1, pcd2_centroid,
                                                                              pcd2_scale)
    pcd2_pred_path2_origin = geometry_utils.denormalize_geometry_tensor_batch(pcd2_pred_path2, pcd2_centroid,
                                                                              pcd2_scale)

    loss_dense_pcd1_path1 = cd_loss_L1(pcd1_pred_path1_origin, pcd1_gt_origin)
    loss_dense_pcd1_path2 = cd_loss_L1(pcd1_pred_path2_origin, pcd1_gt_origin)
    loss_dense_pcd2_path1 = cd_loss_L1(pcd2_pred_path1_origin, pcd2_gt_origin)
    loss_dense_pcd2_path2 = cd_loss_L1(pcd2_pred_path2_origin, pcd2_gt_origin)
    loss_dense = (loss_dense_pcd1_path1 + loss_dense_pcd1_path2 + loss_dense_pcd2_path1 + loss_dense_pcd2_path2) / 4

    loss_surface_pcd1_path1 = medial_axis_surface_loss(center1, radius1, pcd1_pred_path1_origin)
    loss_surface_pcd1_path2 = medial_axis_surface_loss(center1, radius1, pcd1_pred_path2_origin)
    loss_surface_pcd2_path1 = medial_axis_surface_loss(center2, radius2, pcd2_pred_path1_origin)
    loss_surface_pcd2_path2 = medial_axis_surface_loss(center2, radius2, pcd2_pred_path2_origin)
    loss_medial_axis_surface = (loss_surface_pcd1_path1 + loss_surface_pcd1_path2 + loss_surface_pcd2_path1 + loss_surface_pcd2_path2) / 4

    loss_interaction_pcd1_path1 = medial_axis_interaction_loss(center1, radius1, pcd1_pred_path1_origin)
    loss_interaction_pcd1_path2 = medial_axis_interaction_loss(center1, radius1, pcd1_pred_path2_origin)
    loss_interaction_pcd2_path1 = medial_axis_interaction_loss(center2, radius2, pcd2_pred_path1_origin)
    loss_interaction_pcd2_path2 = medial_axis_interaction_loss(center2, radius2, pcd2_pred_path2_origin)
    loss_medial_axis_interaction = (loss_interaction_pcd1_path1 + loss_interaction_pcd1_path2 + loss_interaction_pcd2_path1 + loss_interaction_pcd2_path2) / 4

    loss_ibs_angle_pcd1_path1, intersect_num_pcd1_path1 = ibs_angle_loss(center1, radius1, direction1,
                                                                         pcd1_pred_path1_origin)
    loss_ibs_angle_pcd1_path2, intersect_num_pcd1_path2 = ibs_angle_loss(center1, radius1, direction1,
                                                                         pcd1_pred_path2_origin)
    loss_ibs_angle_pcd2_path1, intersect_num_pcd2_path1 = ibs_angle_loss(center2, radius2, direction2,
                                                                         pcd2_pred_path1_origin)
    loss_ibs_angle_pcd2_path2, intersect_num_pcd2_path2 = ibs_angle_loss(center2, radius2, direction2,
                                                                         pcd2_pred_path2_origin)
    loss_ibs_angle = (loss_ibs_angle_pcd1_path1 + loss_ibs_angle_pcd1_path2 + loss_ibs_angle_pcd2_path1 + loss_ibs_angle_pcd2_path2) / 4
    intersect_num = (intersect_num_pcd1_path1 + intersect_num_pcd1_path2 + intersect_num_pcd2_path1 + intersect_num_pcd2_path2) / 4

    return loss_dense, loss_medial_axis_surface, loss_medial_axis_interaction, loss_ibs_angle, intersect_num


def train(network, train_dataloader, optimizer, epoch, specs, tensorboard_writer):
    device = specs.get("Device")

    network_path1, network_path2 = network
    optimizer_path1, optimizer_path2 = optimizer

    network_path1.train()
    network_path2.train()

    logger.info("")
    logger.info('epoch: {}, path1 learning rate: {}'.format(epoch, optimizer_path1.param_groups[0]["lr"]))
    logger.info('epoch: {}, path2 learning rate: {}'.format(epoch, optimizer_path2.param_groups[0]["lr"]))

    train_total_loss_dense = 0
    train_total_loss_medial_axis_surface = 0
    train_total_loss_medial_axis_interaction = 0
    train_total_loss_ibs_angle = 0
    train_total_intersect_num = 0
    for data, idx in train_dataloader:
        pcd_partial, pcd_gt, pcd_normalize_para, medial_axis_sphere = data
        pcd1_partial, pcd2_partial = pcd_partial
        pcd1_gt, pcd2_gt = pcd_gt
        optimizer_path1.zero_grad()
        optimizer_path2.zero_grad()

        pcd1_partial = pcd1_partial.to(device).permute(0, 2, 1)
        pcd2_partial = pcd2_partial.to(device).permute(0, 2, 1)
        pcd1_gt = pcd1_gt.to(device)
        pcd2_gt = pcd2_gt.to(device)

        pcd_input = torch.concatenate((pcd1_partial, pcd2_partial), dim=2)
        pcd1_pred_path1, pcd2_pred_path1 = network_path1(pcd_input)
        pcd2_pred_path2, pcd1_pred_path2 = network_path2(pcd_input)

        loss_consistency = emd_loss(pcd1_pred_path1, pcd1_pred_path2) + emd_loss(pcd2_pred_path1, pcd2_pred_path2)
        loss_pcd1_path1 = emd_loss(pcd1_pred_path1, pcd1_gt)
        loss_pcd2_path1 = emd_loss(pcd2_pred_path1, pcd2_gt)
        loss_pcd1_path2 = emd_loss(pcd1_pred_path2, pcd1_gt)
        loss_pcd2_path2 = emd_loss(pcd2_pred_path2, pcd2_gt)
        loss_path1 = loss_pcd1_path1 + loss_pcd2_path1
        loss_path2 = loss_pcd1_path2 + loss_pcd2_path2
        total_loss = 0.00001 * (loss_path1 + loss_path2) + loss_consistency

        total_loss.backward()
        optimizer_path1.step()
        optimizer_path2.step()

        pcd_pred = ((pcd1_pred_path1, pcd2_pred_path1), (pcd1_pred_path2, pcd2_pred_path2))
        pcd_gt = (pcd1_gt, pcd2_gt)
        pcd1_normalize_para, pcd2_normalize_para = pcd_normalize_para
        pcd1_centroid, pcd1_scale = pcd1_normalize_para
        pcd2_centroid, pcd2_scale = pcd2_normalize_para
        pcd_normalize_para = ((pcd1_centroid.to(device), pcd1_scale.to(device)), (pcd2_centroid.to(device), pcd2_scale.to(device)))
        medial_axis_sphere1, medial_axis_sphere2 = medial_axis_sphere
        center1, radius1, direction1 = medial_axis_sphere1
        center2, radius2, direction2 = medial_axis_sphere2
        medial_axis_sphere = ((center1.to(device), radius1.to(device), direction1.to(device)), (center2.to(device), radius2.to(device), direction2.to(device)))
        loss_dense, loss_medial_axis_surface, loss_medial_axis_interaction, loss_ibs_angle, intersect_num = get_evaluation_metrics(pcd_pred, pcd_gt, pcd_normalize_para, medial_axis_sphere)

        train_total_loss_dense += loss_dense.item()
        train_total_loss_medial_axis_surface += loss_medial_axis_surface.item()
        train_total_loss_medial_axis_interaction += loss_medial_axis_interaction.item()
        train_total_loss_ibs_angle += loss_ibs_angle.item()
        train_total_intersect_num += intersect_num.item()

    record_loss_info(specs, "train_loss_dense", train_total_loss_dense / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_medial_axis_surface", train_total_loss_medial_axis_surface / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_medial_axis_interaction", train_total_loss_medial_axis_interaction / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_loss_ibs_angle", train_total_loss_ibs_angle / train_dataloader.__len__(), epoch, tensorboard_writer)
    record_loss_info(specs, "train_intersect_num", train_total_intersect_num / train_dataloader.__len__(), epoch, tensorboard_writer)


def test(network, test_dataloader, epoch, specs, tensorboard_writer, best_cd, best_epoch):
    logger = LogFactory.get_logger(specs.get("LogOptions"))
    device = specs.get("Device")

    network_path1, network_path2 = network

    network_path1.eval()
    network_path2.eval()
    with torch.no_grad():
        test_total_dense = 0
        test_total_medial_axis_surface = 0
        test_total_medial_axis_interaction = 0
        test_total_ibs_angle = 0
        test_total_intersect_num = 0
        for data, idx in test_dataloader:
            pcd_partial, pcd_gt, pcd_normalize_para, medial_axis_sphere = data
            pcd1_partial, pcd2_partial = pcd_partial
            pcd1_gt, pcd2_gt = pcd_gt

            pcd1_partial = pcd1_partial.to(device).permute(0, 2, 1)
            pcd2_partial = pcd2_partial.to(device).permute(0, 2, 1)
            pcd1_gt = pcd1_gt.to(device)
            pcd2_gt = pcd2_gt.to(device)

            pcd1_pred_path1, pcd2_pred_path1 = network_path1(torch.concatenate((pcd1_partial, pcd2_partial), dim=2))
            pcd2_pred_path2, pcd1_pred_path2 = network_path2(torch.concatenate((pcd1_partial, pcd2_partial), dim=2))

            pcd_pred = ((pcd1_pred_path1, pcd2_pred_path1), (pcd1_pred_path2, pcd2_pred_path2))
            pcd_gt = (pcd1_gt, pcd2_gt)
            pcd1_normalize_para, pcd2_normalize_para = pcd_normalize_para
            pcd1_centroid, pcd1_scale = pcd1_normalize_para
            pcd2_centroid, pcd2_scale = pcd2_normalize_para
            pcd_normalize_para = ((pcd1_centroid.to(device), pcd1_scale.to(device)), (pcd2_centroid.to(device), pcd2_scale.to(device)))
            medial_axis_sphere1, medial_axis_sphere2 = medial_axis_sphere
            center1, radius1, direction1 = medial_axis_sphere1
            center2, radius2, direction2 = medial_axis_sphere2
            medial_axis_sphere = ((center1.to(device), radius1.to(device), direction1.to(device)), (center2.to(device), radius2.to(device), direction2.to(device)))
            loss_dense, loss_medial_axis_surface, loss_medial_axis_interaction, loss_ibs_angle, intersect_num = get_evaluation_metrics(pcd_pred, pcd_gt, pcd_normalize_para, medial_axis_sphere)

            test_total_dense += loss_dense.item()
            test_total_medial_axis_surface += loss_medial_axis_surface.item()
            test_total_medial_axis_interaction += loss_medial_axis_interaction.item()
            test_total_ibs_angle += loss_ibs_angle.item()
            test_total_intersect_num += intersect_num.item()

        test_avrg_dense = test_total_dense / test_dataloader.__len__()
        record_loss_info(specs, "test_loss_dense", test_total_dense / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_medial_axis_surface", test_total_medial_axis_surface / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_medial_axis_interaction", test_total_medial_axis_interaction / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_loss_ibs_angle", test_total_ibs_angle / test_dataloader.__len__(), epoch, tensorboard_writer)
        record_loss_info(specs, "test_intersect_num", test_total_intersect_num / test_dataloader.__len__(), epoch, tensorboard_writer)

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

    train_loader, test_loader = get_dataloader(dataset_RBPCDC.RBPCDCDataset, specs)
    network_path1 = get_network(specs, TopNet_path1, None, input_num=2048)
    network_path2 = get_network(specs, TopNet_path2, None, input_num=2048)
    optimizer_path1 = get_optimizer(specs, network_path1, None)
    optimizer_path2 = get_optimizer(specs, network_path2, None)
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
        network = (network_path1, network_path2)
        optimizer = (optimizer_path1, optimizer_path2)

        train(network, train_loader, optimizer, epoch, specs, tensorboard_writer)
        time_end_train = time.time()
        logger.info("use {} to train".format(time_end_train - time_begin_train))

        time_begin_test = time.time()
        best_cd, best_epoch = test(network, test_loader, epoch, specs, tensorboard_writer, best_cd, best_epoch)
        time_end_test = time.time()
        logger.info("use {} to test".format(time_end_test - time_begin_test))

    tensorboard_writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train IBPCDC")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/INTE/train/specs_train_RBPCDC_INTE.json",
        required=False,
        help="The experiment config file."
    )

    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)

    logger = LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("specs file path: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))

    main_function(specs)
