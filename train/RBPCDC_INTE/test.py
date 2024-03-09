import sys
import os

sys.path.insert(0, "/home/data/jinshuo/IBPCDC")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import time
import os.path
from datetime import datetime, timedelta

from models.RBPCDC import TopNet_path1, TopNet_path2
from utils.metric import *
from utils.test_utils import *
from utils import log_utils, path_utils, statistics_utils, geometry_utils
from dataset import dataset_RBPCDC


def save_result(filename_list: list, pcd, specs: dict):
    pcd_pred_path1, pcd_pred_path2 = pcd
    pcd1_pred_path1, pcd2_pred_path1 = pcd_pred_path1
    pcd1_pred_path2, pcd2_pred_path2 = pcd_pred_path2

    save_dir = specs.get("ResultSaveDir")
    tag = specs.get("TAG")
    dataset = specs.get("Dataset")
    normalize_para_dir = specs.get("NormalizeParaDir")

    filename_patten = specs.get("FileNamePatten")
    scene_patten = specs.get("ScenePatten")

    pcd1_pred_path1 = pcd1_pred_path1.cpu().detach().numpy()
    pcd2_pred_path1 = pcd2_pred_path1.cpu().detach().numpy()
    pcd1_pred_path2 = pcd1_pred_path2.cpu().detach().numpy()
    pcd2_pred_path2 = pcd2_pred_path2.cpu().detach().numpy()

    for index, filename_abs in enumerate(filename_list):
        # [dataset, category, filename], example:[MVP, scene1, scene1.1000_view0_0.ply]
        _, category, filename = filename_abs.split('/')
        filename = re.match(filename_patten, filename).group()  # scene1.1000_view0_0

        # the real directory is save_dir/tag/dataset/category
        path1_save_path = os.path.join(save_dir, tag, dataset, "path1", category)
        path2_save_path = os.path.join(save_dir, tag, dataset, "path1", category)

        if not os.path.isdir(path1_save_path):
            os.makedirs(path1_save_path)
        if not os.path.isdir(path2_save_path):
            os.makedirs(path2_save_path)

        # get final filename
        filename_final = "{}.ply".format(filename)
        absolute_path = os.path.join(save_path, filename_final)
        pcd = get_pcd_from_np(pcd_np[index])

        # transform to origin coordinate
        pcd.scale(scale, np.array([0, 0, 0]))
        pcd.translate(translate)

        o3d.io.write_point_cloud(absolute_path, pcd)


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

    cd_l1_pcd1_path1 = l1_cd(pcd1_pred_path1_origin, pcd1_gt_origin)
    cd_l1_pcd1_path2 = l1_cd(pcd1_pred_path2_origin, pcd1_gt_origin)
    cd_l1_pcd2_path1 = l1_cd(pcd2_pred_path1_origin, pcd2_gt_origin)
    cd_l1_pcd2_path2 = l1_cd(pcd2_pred_path2_origin, pcd2_gt_origin)
    cd_l1 = (cd_l1_pcd1_path1 + cd_l1_pcd1_path2 + cd_l1_pcd2_path1 + cd_l1_pcd2_path2) / 4

    emd_pcd1_path1 = emd(pcd1_pred_path1_origin, pcd1_gt_origin)
    emd_pcd1_path2 = emd(pcd1_pred_path2_origin, pcd1_gt_origin)
    emd_pcd2_path1 = emd(pcd2_pred_path1_origin, pcd2_gt_origin)
    emd_pcd2_path2 = emd(pcd2_pred_path2_origin, pcd2_gt_origin)
    emd_ = (emd_pcd1_path1 + emd_pcd1_path2 + emd_pcd2_path1 + emd_pcd2_path2) / 4

    f_score_pcd1_path1 = f_score(pcd1_pred_path1_origin, pcd1_gt_origin)
    f_score_pcd1_path2 = f_score(pcd1_pred_path2_origin, pcd1_gt_origin)
    f_score_pcd2_path1 = f_score(pcd2_pred_path1_origin, pcd2_gt_origin)
    f_score_pcd2_path2 = f_score(pcd2_pred_path2_origin, pcd2_gt_origin)
    fscore = (f_score_pcd1_path1 + f_score_pcd1_path2 + f_score_pcd2_path1 + f_score_pcd2_path2) / 4

    mad_s_pcd1_path1 = medial_axis_surface_dist(center1, radius1, pcd1_pred_path1_origin)
    mad_s_pcd1_path2 = medial_axis_surface_dist(center1, radius1, pcd1_pred_path2_origin)
    mad_s_pcd2_path1 = medial_axis_surface_dist(center2, radius2, pcd2_pred_path1_origin)
    mad_s_pcd2_path2 = medial_axis_surface_dist(center2, radius2, pcd2_pred_path2_origin)
    mad_s = (mad_s_pcd1_path1 + mad_s_pcd1_path2 + mad_s_pcd2_path1 + mad_s_pcd2_path2) / 4

    mad_i_pcd1_path1 = medial_axis_interaction_dist(center1, radius1, pcd1_pred_path1_origin)
    mad_i_pcd1_path2 = medial_axis_interaction_dist(center1, radius1, pcd1_pred_path2_origin)
    mad_i_pcd2_path1 = medial_axis_interaction_dist(center2, radius2, pcd2_pred_path1_origin)
    mad_i_pcd2_path2 = medial_axis_interaction_dist(center2, radius2, pcd2_pred_path2_origin)
    mad_i = (mad_i_pcd1_path1 + mad_i_pcd1_path2 + mad_i_pcd2_path1 + mad_i_pcd2_path2) / 4

    ibs_a_pcd1_path1, interact_num_pcd1_path1 = ibs_angle_dist(center1, radius1, direction1,
                                                                         pcd1_pred_path1_origin)
    ibs_a_pcd1_path2, interact_num_pcd1_path2 = ibs_angle_dist(center1, radius1, direction1,
                                                                         pcd1_pred_path2_origin)
    ibs_a_pcd2_path1, interact_num_pcd2_path1 = ibs_angle_dist(center2, radius2, direction2,
                                                                         pcd2_pred_path1_origin)
    ibs_a_pcd2_path2, interact_num_pcd2_path2 = ibs_angle_dist(center2, radius2, direction2,
                                                                         pcd2_pred_path2_origin)
    ibs_a = (ibs_a_pcd1_path1 + ibs_a_pcd1_path2 + ibs_a_pcd2_path1 + ibs_a_pcd2_path2) / 4
    interact_num = (interact_num_pcd1_path1 + interact_num_pcd1_path2 + interact_num_pcd2_path1 + interact_num_pcd2_path2) / 4

    return cd_l1, emd_, fscore, mad_s, mad_i, ibs_a, interact_num


def test(model, test_dataloader, specs):
    network_path1, network_path2 = model
    device = specs.get("Device")

    dist_dict = {
        "cd_l1": {},
        "emd": {},
        "fscore": {},
        "mad_s": {},
        "mad_i": {},
        "ibs_a": {},
        "interact_num": {}
    }
    single_csv_data = {}
    model.eval()
    with torch.no_grad():
        for data, idx in test_dataloader:
            filename_list = [test_dataloader.dataset.pcd1_partial_filenames[i] for i in idx]
            pcd_partial, pcd_gt, pcd_normalize_para, medial_axis_sphere = data

            pcd1_partial, pcd2_partial = pcd_partial
            pcd1_gt, pcd2_gt = pcd_gt

            pcd_partial = pcd_partial.to(device)
            pcd_gt = pcd_gt.to(device)
            center = center.to(device)
            radius = radius.to(device)
            direction = direction.to(device)

            pcd_input = torch.concatenate((pcd1_partial, pcd2_partial), dim=2)
            pcd1_pred_path1, pcd2_pred_path1 = network_path1(pcd_input)
            pcd2_pred_path2, pcd1_pred_path2 = network_path2(pcd_input)

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
            cd_l1, emd_, fscore, mad_s, mad_i, ibs_a, interact_num = get_evaluation_metrics(pcd_pred, pcd_gt, pcd_normalize_para, medial_axis_sphere)

            update_loss_dict(dist_dict, filename_list, cd_l1.detach().cpu().numpy(), "cd_l1")
            update_loss_dict(dist_dict, filename_list, emd_.detach().cpu().numpy(), "emd")
            update_loss_dict(dist_dict, filename_list, fscore.detach().cpu().numpy(), "fscore")
            update_loss_dict(dist_dict, filename_list, mad_s.detach().cpu().numpy(), "mad_s")
            update_loss_dict(dist_dict, filename_list, mad_i.detach().cpu().numpy(), "mad_i")
            update_loss_dict(dist_dict, filename_list, ibs_a.detach().cpu().numpy(), "ibs_a")
            update_loss_dict(dist_dict, filename_list, interact_num.detach().cpu().numpy(), "interact_num")

            statistics_utils.append_csv_data(single_csv_data, filename_list, cd_l1, emd_, fscore, mad_s, mad_i, ibs_a, interact_num)

            save_result(filename_list, pcd_pred, specs)
            logger.info("saved {} pcds".format(idx.shape[0]))

        cal_avrg_dist(dist_dict)
        logger.info("dist result: \n{}".format(json.dumps(dist_dict, sort_keys=False, indent=4)))

        csv_file_dir = os.path.join(specs.get("ResultSaveDir"), specs.get("TAG"), specs.get("Dataset"))
        avrg_csv_file_path = os.path.join(csv_file_dir, "avrg_result.csv")
        statistics_utils.write_avrg_csv_file(avrg_csv_file_path, dist_dict, "INTE")

        single_csv_file_path = os.path.join(csv_file_dir, "single_result.csv")
        statistics_utils.write_single_csv_file(single_csv_file_path, single_csv_data)


def main_function(specs):
    device = specs.get("Device")
    model_path = specs.get("ModelPath")
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    logger.info("test device: {}".format(device))
    logger.info("batch size: {}".format(specs.get("BatchSize")))

    test_dataloader = get_dataloader(dataset_RBPCDC.RBPCDCDataset, specs)
    logger.info("init dataloader succeed")

    network_path1 = get_network(specs, TopNet_path1, None, input_num=2048)
    network_path2 = get_network(specs, TopNet_path2, None, input_num=2048)
    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    network_path1.load_state_dict(checkpoint["model_path1"])
    network_path2.load_state_dict(checkpoint["model_path2"])
    logger.info("load trained model succeed, epoch: {}".format(checkpoint["epoch"]))

    time_begin_test = time.time()
    model = (network_path1, network_path2)
    test(model, test_dataloader, specs)
    time_end_test = time.time()
    logger.info("use {} to test".format(time_end_test - time_begin_test))

    time_begin_zip = time.time()
    create_zip(specs)
    time_end_zip = time.time()
    logger.info("use {} to zip".format(time_end_zip - time_begin_zip))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_config_file",
        default="configs/INTE/test/specs_test_TopNet_INTE.json",
        required=False,
        help="The experiment config file."
    )
    args = arg_parser.parse_args()

    specs = path_utils.read_config(args.experiment_config_file)
    logger = log_utils.LogFactory.get_logger(specs.get("LogOptions"))
    
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))
    logger.info("current time: {}".format(TIMESTAMP))
    logger.info("test split: {}".format(specs.get("TestSplit")))
    logger.info("specs file: {}".format(args.experiment_config_file))
    logger.info("specs file: \n{}".format(json.dumps(specs, sort_keys=False, indent=4)))
    logger.info("model: {}".format(specs.get("ModelPath")))

    main_function(specs)
