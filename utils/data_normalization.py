import logging
import numpy as np
import os
import torch
import torch.utils.data
import open3d as o3d
import re

import utils.workspace as ws


def get_instance_filenames(data_source, split):
    IBS_filenames = []
    pcd_partial_filenames = []
    pcd_gt_filenames = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                scene_name = re.match(ws.scene_patten, instance_name).group()
                IBS1_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(scene_name, 0))
                IBS2_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(scene_name, 1))
                pcd1_partial_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(instance_name, 0))
                pcd2_partial_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(instance_name, 1))
                pcd1_gt_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(scene_name, 0))
                pcd2_gt_filename = os.path.join(dataset, class_name, "{}_{}.ply".format(scene_name, 1))

                if not os.path.isfile(os.path.join(data_source, ws.IBS_gt_subdir, IBS1_filename)):
                    logging.warning("Requested non-existent file '{}'".format(IBS1_filename))

                IBS_filenames.append(IBS1_filename)
                IBS_filenames.append(IBS2_filename)
                pcd_partial_filenames.append(pcd1_partial_filename)
                pcd_partial_filenames.append(pcd2_partial_filename)
                pcd_gt_filenames.append(pcd1_gt_filename)
                pcd_gt_filenames.append(pcd2_gt_filename)

    return IBS_filenames, pcd_partial_filenames, pcd_gt_filenames


def get_pcd_data(pcd_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    xyz_load = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
    return xyz_load


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, split):
        self.data_source = data_source
        self.IBS_filenames, self.pcd_partial_filenames, self.pcd_gt_filenames = get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.IBS_filenames)

    def __getitem__(self, idx):
        IBS_filename = os.path.join(self.data_source, ws.IBS_gt_subdir, self.IBS_filenames[idx])
        pcd_partial_filename = os.path.join(self.data_source, ws.pcd_partial_subdir, self.pcd_partial_filenames[idx])
        pcd1_gt_filename = os.path.join(self.data_source, ws.pcd_complete_subdir, self.pcd_gt_filenames[idx])

        IBS = get_pcd_data(IBS_filename)
        pcd_partial = get_pcd_data(pcd_partial_filename)
        pcd_gt = get_pcd_data(pcd1_gt_filename)

        return IBS, pcd_partial, pcd_gt, idx
