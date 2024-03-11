import logging
import numpy as np
import os
import torch
import torch.utils.data
import open3d as o3d
import re

import utils.workspace as ws


def get_instance_filenames(data_source, split):
    pcd1_partial_filenames = []
    pcd2_partial_filenames = []
    pcd1_gt_filenames = []
    pcd2_gt_filenames = []
    pcd1_normalize_para_filenames = []
    pcd2_normalize_para_filenames = []
    medial_axis_sphere1_filenames = []
    medial_axis_sphere2_filenames = []

    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                scene_name = re.match(ws.scene_patten, instance_name).group()
                pcd1_partial_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(instance_name, 0))
                pcd2_partial_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(instance_name, 1))
                pcd1_gt_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(scene_name, 0))
                pcd2_gt_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(scene_name, 1))
                pcd1_normalize_para_filename = os.path.join(dataset, class_name, "{}_{}.txt".format(scene_name, 0))
                pcd2_normalize_para_filename = os.path.join(dataset, class_name, "{}_{}.txt".format(scene_name, 1))
                medial_axis_sphere1_filename = os.path.join(dataset, class_name, "{}_{}.npz".format(scene_name, 0))
                medial_axis_sphere2_filename = os.path.join(dataset, class_name, "{}_{}.npz".format(scene_name, 1))

                if not os.path.isfile(os.path.join(data_source, ws.medial_axis_sphere_subdir, medial_axis_sphere1_filename)):
                    logging.warning("Requested non-existent file '{}'".format(pcd1_partial_filename))

                pcd1_partial_filenames.append(pcd1_partial_filename)
                pcd2_partial_filenames.append(pcd2_partial_filename)
                pcd1_gt_filenames.append(pcd1_gt_filename)
                pcd2_gt_filenames.append(pcd2_gt_filename)
                pcd1_normalize_para_filenames.append(pcd1_normalize_para_filename)
                pcd2_normalize_para_filenames.append(pcd2_normalize_para_filename)
                medial_axis_sphere1_filenames.append(medial_axis_sphere1_filename)
                medial_axis_sphere2_filenames.append(medial_axis_sphere2_filename)

    pcd_partial_filenames = (pcd1_partial_filenames, pcd2_partial_filenames)
    pcd_gt_filenames = (pcd1_gt_filenames, pcd2_gt_filenames)
    pcd_normalize_para_filenames = (pcd1_normalize_para_filenames, pcd2_normalize_para_filenames)
    medial_axis_sphere_filenames = (medial_axis_sphere1_filenames, medial_axis_sphere2_filenames)

    return pcd_partial_filenames, pcd_gt_filenames, pcd_normalize_para_filenames, medial_axis_sphere_filenames


def get_pcd_data(pcd_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    xyz_load = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
    return xyz_load


def _get_normalize_para(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        data = list(map(float, content.split(",")))
        translate = data[0:3]
        scale = data[-1]
    return translate, scale


def get_normalize_para_tensor(file_path):
    translate, scale = _get_normalize_para(file_path)
    return torch.tensor(translate, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32)


def get_normalize_para_np(file_path):
    translate, scale = _get_normalize_para(file_path)
    return np.asarray(translate, dtype=np.float32), np.asarray(scale, dtype=np.float32)


def get_medial_axis_sphere_data(medial_axis_sphere_filename):
    data = np.load(medial_axis_sphere_filename)
    center = data["center"]
    radius = data["radius"]
    direction = data["direction"]
    center = torch.from_numpy(np.asarray(center).astype(np.float32))
    radius = torch.from_numpy(np.asarray(radius).astype(np.float32))
    direction = torch.from_numpy(np.asarray(direction).astype(np.float32))
    return center, radius, direction


class RBPCDCDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, split):
        self.data_source = data_source
        pcd_partial_filenames, pcd_gt_filenames, pcd_normalize_para_filenames, medial_axis_sphere_filenames = get_instance_filenames(data_source, split)
        self.pcd1_partial_filenames, self.pcd2_partial_filenames = pcd_partial_filenames
        self.pcd1_gt_filenames, self.pcd2_gt_filenames = pcd_gt_filenames
        self.pcd1_normalize_para_filenames, self.pcd2_normalize_para_filenames = pcd_normalize_para_filenames
        self.medial_axis_sphere1_filenames, self.medial_axis_sphere2_filenames = medial_axis_sphere_filenames

    def __len__(self):
        return len(self.pcd1_partial_filenames)

    def __getitem__(self, idx):
        pcd1_partial_filename = os.path.join(self.data_source, ws.pcd_partial_subdir, self.pcd1_partial_filenames[idx])
        pcd2_partial_filename = os.path.join(self.data_source, ws.pcd_partial_subdir, self.pcd1_partial_filenames[idx])
        pcd1_gt_filename = os.path.join(self.data_source, ws.pcd_complete_subdir, self.pcd1_gt_filenames[idx])
        pcd2_gt_filename = os.path.join(self.data_source, ws.pcd_complete_subdir, self.pcd2_gt_filenames[idx])
        pcd1_normalize_data_filename = os.path.join(self.data_source, ws.normalize_para_subdir, self.pcd1_normalize_para_filenames[idx])
        pcd2_normalize_data_filename = os.path.join(self.data_source, ws.normalize_para_subdir, self.pcd2_normalize_para_filenames[idx])
        medial_axis_sphere1_filename = os.path.join(self.data_source, ws.medial_axis_sphere_subdir, self.medial_axis_sphere1_filenames[idx])
        medial_axis_sphere2_filename = os.path.join(self.data_source, ws.medial_axis_sphere_subdir, self.medial_axis_sphere2_filenames[idx])

        pcd_partial = (get_pcd_data(pcd1_partial_filename), get_pcd_data(pcd2_partial_filename))
        pcd_gt = (get_pcd_data(pcd1_gt_filename), get_pcd_data(pcd2_gt_filename))
        pcd_normalize_para = (get_normalize_para_tensor(pcd1_normalize_data_filename), get_normalize_para_tensor(pcd2_normalize_data_filename))
        medial_axis_sphere = (get_medial_axis_sphere_data(medial_axis_sphere1_filename), get_medial_axis_sphere_data(medial_axis_sphere2_filename))

        return (pcd_partial, pcd_gt, pcd_normalize_para, medial_axis_sphere), idx
