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
from dataset.dataset_RBPCDC import get_normalize_para_np


def get_network(specs, model_class, checkpoint, model_tag, **kwargs):
    assert checkpoint is not None

    device = specs.get("Device")
    logger = LogFactory.get_logger(specs.get("LogOptions"))

    network = model_class(**kwargs).to(device)

    logger.info("load model parameter from epoch {}".format(checkpoint["epoch"]))
    network.load_state_dict(checkpoint[model_tag])

    return network


def save_pcd(pcd, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path)
    o3d.io.write_point_cloud(os.path.join(path, filename), pcd)


def save_result(filename_list: list, pcd, specs: dict):
    filename_list = filename_list[0: int(len(filename_list)/2)]
    pcd_pred_path1, pcd_pred_path2 = pcd
    pcd1_pred_path1, pcd2_pred_path1 = pcd_pred_path1
    pcd1_pred_path2, pcd2_pred_path2 = pcd_pred_path2

    save_dir = specs.get("ResultSaveDir")
    tag = specs.get("TAG")
    dataset = specs.get("Dataset")

    viewname_patten = specs.get("ViewNamePatten")
    scene_patten = specs.get("ScenePatten")

    pcd1_pred_path1_batch = pcd1_pred_path1.cpu().detach().numpy()
    pcd1_pred_path2_batch = pcd1_pred_path2.cpu().detach().numpy()
    pcd2_pred_path1_batch = pcd2_pred_path1.cpu().detach().numpy()
    pcd2_pred_path2_batch = pcd2_pred_path2.cpu().detach().numpy()

    for index, filename_abs in enumerate(filename_list):
        # [dataset, category, filename], example:[MVP, scene1, scene1.1000_view0_0.ply]
        _, category, filename = filename_abs.split('/')
        filename = re.match(viewname_patten, filename).group()  # scene1.1000_view0
        scene = re.match(scene_patten, filename).group()  # scene1.1000

        # the save directory is save_dir/tag/dataset/path{0, 1}/category
        path1_save_path = os.path.join(save_dir, tag, dataset, "path1", category)
        path2_save_path = os.path.join(save_dir, tag, dataset, "path2", category)

        # get final filename
        pcd1_filename_final = "{}_0.ply".format(filename)
        pcd2_filename_final = "{}_1.ply".format(filename)

        # normalize every geometry
        pcd1_pred_path1 = get_pcd_from_np(pcd1_pred_path1_batch[index])
        pcd1_pred_path2 = get_pcd_from_np(pcd1_pred_path2_batch[index])
        pcd2_pred_path1 = get_pcd_from_np(pcd2_pred_path1_batch[index])
        pcd2_pred_path2 = get_pcd_from_np(pcd2_pred_path2_batch[index])

        save_pcd(pcd1_pred_path1, path1_save_path, pcd1_filename_final)
        save_pcd(pcd1_pred_path2, path2_save_path, pcd1_filename_final)
        save_pcd(pcd2_pred_path1, path1_save_path, pcd2_filename_final)
        save_pcd(pcd2_pred_path2, path2_save_path, pcd2_filename_final)


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

    pcd1_gt_origin = geometry_utils.normalize_geometry_tensor_batch(pcd1_gt, pcd1_centroid, pcd1_scale)
    pcd1_pred_path1_origin = geometry_utils.normalize_geometry_tensor_batch(pcd1_pred_path1, pcd1_centroid, pcd1_scale)
    pcd1_pred_path2_origin = geometry_utils.normalize_geometry_tensor_batch(pcd1_pred_path2, pcd1_centroid, pcd1_scale)
    pcd2_gt_origin = geometry_utils.normalize_geometry_tensor_batch(pcd2_gt, pcd2_centroid, pcd2_scale)
    pcd2_pred_path1_origin = geometry_utils.normalize_geometry_tensor_batch(pcd2_pred_path1, pcd2_centroid, pcd2_scale)
    pcd2_pred_path2_origin = geometry_utils.normalize_geometry_tensor_batch(pcd2_pred_path2, pcd2_centroid, pcd2_scale)

    cd_l1_pcd1_path1 = l1_cd(pcd1_pred_path1_origin, pcd1_gt_origin)
    cd_l1_pcd1_path2 = l1_cd(pcd1_pred_path2_origin, pcd1_gt_origin)
    cd_l1_pcd2_path1 = l1_cd(pcd2_pred_path1_origin, pcd2_gt_origin)
    cd_l1_pcd2_path2 = l1_cd(pcd2_pred_path2_origin, pcd2_gt_origin)
    cd_l1_pcd1 = (cd_l1_pcd1_path1 + cd_l1_pcd1_path2) / 2
    cd_l1_pcd2 = (cd_l1_pcd2_path1 + cd_l1_pcd2_path2) / 2
    cd_l1 = torch.cat((cd_l1_pcd1, cd_l1_pcd2), dim=0)

    emd_pcd1_path1 = emd(pcd1_pred_path1_origin, pcd1_gt_origin)
    emd_pcd1_path2 = emd(pcd1_pred_path2_origin, pcd1_gt_origin)
    emd_pcd2_path1 = emd(pcd2_pred_path1_origin, pcd2_gt_origin)
    emd_pcd2_path2 = emd(pcd2_pred_path2_origin, pcd2_gt_origin)
    emd_pcd1 = (emd_pcd1_path1 + emd_pcd1_path2) / 2
    emd_pcd2 = (emd_pcd2_path1 + emd_pcd2_path2) / 2
    emd_ = torch.cat((emd_pcd1, emd_pcd2), dim=0)

    f_score_pcd1_path1 = f_score(pcd1_pred_path1_origin, pcd1_gt_origin)
    f_score_pcd1_path2 = f_score(pcd1_pred_path2_origin, pcd1_gt_origin)
    f_score_pcd2_path1 = f_score(pcd2_pred_path1_origin, pcd2_gt_origin)
    f_score_pcd2_path2 = f_score(pcd2_pred_path2_origin, pcd2_gt_origin)
    f_score_pcd1 = (f_score_pcd1_path1 + f_score_pcd1_path2) / 2
    f_score_pcd2 = (f_score_pcd2_path1 + f_score_pcd2_path2) / 2
    f_score_ = torch.cat((f_score_pcd1, f_score_pcd2), dim=0)

    mas_pcd1_path1 = medial_axis_surface_dist(center1, radius1, pcd1_pred_path1_origin)
    mas_pcd1_path2 = medial_axis_surface_dist(center1, radius1, pcd1_pred_path2_origin)
    mas_pcd2_path1 = medial_axis_surface_dist(center2, radius2, pcd2_pred_path1_origin)
    mas_pcd2_path2 = medial_axis_surface_dist(center2, radius2, pcd2_pred_path2_origin)
    mas_pcd1 = (mas_pcd1_path1 + mas_pcd1_path2) / 2
    mas_pcd2 = (mas_pcd2_path1 + mas_pcd2_path2) / 2
    mas_ = torch.cat((mas_pcd1, mas_pcd2), dim=0)

    mai_pcd1_path1 = medial_axis_interaction_dist(center1, radius1, pcd1_pred_path1_origin)
    mai_pcd1_path2 = medial_axis_interaction_dist(center1, radius1, pcd1_pred_path2_origin)
    mai_pcd2_path1 = medial_axis_interaction_dist(center2, radius2, pcd2_pred_path1_origin)
    mai_pcd2_path2 = medial_axis_interaction_dist(center2, radius2, pcd2_pred_path2_origin)
    mai_pcd1 = (mai_pcd1_path1 + mai_pcd1_path2) / 2
    mai_pcd2 = (mai_pcd2_path1 + mai_pcd2_path2) / 2
    mai_ = torch.cat((mai_pcd1, mai_pcd2), dim=0)

    ibss_pcd1_path1, interact_num_pcd1_path1 = ibs_angle_dist(center1, radius1, direction1,
                                                                         pcd1_pred_path1_origin)
    ibss_pcd1_path2, interact_num_pcd1_path2 = ibs_angle_dist(center1, radius1, direction1,
                                                                         pcd1_pred_path2_origin)
    ibss_pcd2_path1, interact_num_pcd2_path1 = ibs_angle_dist(center2, radius2, direction2,
                                                                         pcd2_pred_path1_origin)
    ibss_pcd2_path2, interact_num_pcd2_path2 = ibs_angle_dist(center2, radius2, direction2,
                                                                         pcd2_pred_path2_origin)
    ibss_pcd1 = (ibss_pcd1_path1 + ibss_pcd1_path2) / 2
    ibss_pcd2 = (ibss_pcd2_path1 + ibss_pcd2_path2) / 2
    interact_num_pcd1 = (interact_num_pcd1_path1 + interact_num_pcd1_path2) / 2
    interact_num_pcd2 = (interact_num_pcd2_path1 + interact_num_pcd2_path2) / 2
    ibss_ = torch.cat((ibss_pcd1, ibss_pcd2), dim=0)
    interact_num = torch.cat((interact_num_pcd1, interact_num_pcd2), dim=0)

    return cd_l1, emd_, f_score_, mas_, mai_, ibss_, interact_num


def get_filename_list(specs, idx, test_dataloader):
    viewname_patten = specs.get("ViewNamePatten")
    filename1_list = []
    filename2_list = []
    for i in idx:
        viewname = re.match(".*{}".format(viewname_patten), test_dataloader.dataset.pcd1_partial_filenames[i]).group()
        filename1 = "{}_0.ply".format(viewname)
        filename2 = "{}_1.ply".format(viewname)
        filename1_list.append(filename1)
        filename2_list.append(filename2)
    return filename1_list + filename2_list


def test(model, test_dataloader, specs):
    network_path1, network_path2 = model
    device = specs.get("Device")

    dist_dict = {
        "cd": {},
        "emd": {},
        "fscore": {},
        "mas": {},
        "mai": {},
        "ibss": {},
        "interact_num": {}
    }
    single_csv_data = {}
    network_path1.eval()
    network_path2.eval()
    with torch.no_grad():
        for data, idx in test_dataloader:
            filename_list = get_filename_list(specs, idx, test_dataloader)

            pcd_partial, pcd_gt, pcd_normalize_para, medial_axis_sphere = data

            pcd1_partial, pcd2_partial = pcd_partial
            pcd1_gt, pcd2_gt = pcd_gt

            pcd1_partial = pcd1_partial.to(device).permute(0, 2, 1)
            pcd2_partial = pcd2_partial.to(device).permute(0, 2, 1)
            pcd1_gt = pcd1_gt.to(device)
            pcd2_gt = pcd2_gt.to(device)

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
            cd, emd_, fscore, mas, mai, ibss, interact_num = get_evaluation_metrics(pcd_pred, pcd_gt, pcd_normalize_para, medial_axis_sphere)

            update_loss_dict(dist_dict, filename_list, cd.detach().cpu().numpy(), "cd")
            update_loss_dict(dist_dict, filename_list, emd_.detach().cpu().numpy(), "emd")
            update_loss_dict(dist_dict, filename_list, fscore.detach().cpu().numpy(), "fscore")
            update_loss_dict(dist_dict, filename_list, mas.detach().cpu().numpy(), "mas")
            update_loss_dict(dist_dict, filename_list, mai.detach().cpu().numpy(), "mai")
            update_loss_dict(dist_dict, filename_list, ibss.detach().cpu().numpy(), "ibss")
            update_loss_dict(dist_dict, filename_list, interact_num.detach().cpu().numpy(), "interact_num")

            statistics_utils.append_csv_data(single_csv_data, filename_list, cd, emd_, fscore, mas, mai, ibss, interact_num)

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

    checkpoint = torch.load(model_path, map_location="cuda:{}".format(device))
    network_path1 = get_network(specs, TopNet_path1, checkpoint, "model_path1", input_num=2048)
    network_path2 = get_network(specs, TopNet_path2, checkpoint, "model_path2", input_num=2048)
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
        default="configs/INTE/test/specs_test_RBPCDC_INTE.json",
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
