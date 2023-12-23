#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import open3d as o3d
import re

import utils.workspace as ws


def get_instance_filenames(data_source, split):
    IBSfiles = []
    # partial pcd
    pcd1files = []
    pcd2files = []
    # complete pcd
    pcd1gtfiles = []
    pcd2gtfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                scene_name = re.match(ws.scene_patten, instance_name).group()
                IBS_filename = os.path.join(
                    dataset, class_name, scene_name + ".ply"
                )
                pcd1_filename = os.path.join(
                    dataset, class_name, instance_name + "_0.ply"
                )
                pcd2_filename = os.path.join(
                    dataset, class_name, instance_name + "_1.ply"
                )
                pcd1gt_filename = os.path.join(
                    dataset, class_name, scene_name + "_0.ply"
                )
                pcd2gt_filename = os.path.join(
                    dataset, class_name, scene_name + "_1.ply"
                )
                if not os.path.isfile(
                        os.path.join(data_source, ws.IBS_gt_subdir, IBS_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(IBS_filename)
                    )
                IBSfiles += [IBS_filename]
                pcd1files += [pcd1_filename]
                pcd2files += [pcd2_filename]
                pcd1gtfiles += [pcd1gt_filename]
                pcd2gtfiles += [pcd2gt_filename]

    return IBSfiles, pcd1files, pcd2files, pcd1gtfiles, pcd2gtfiles


def get_pcd_data(pcd_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    xyz_load = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
    return xyz_load


class DataCache:
    def __init__(self, capacity):
        # the max size of the cache
        self._capacity = capacity
        # record the insert order
        self._order = list()
        # cache the file using the format {filename: file}
        self._cache = dict()

    def get(self, filename):
        # 1. get from cache
        file = self._cache.get(filename)

        # 2. not in cache, get from disk, ensure exist, update cache
        if file is None:
            file = get_pcd_data(filename)
            assert file is not None
            self._update_cache(filename, file)

        return file

    def _update_cache(self, filename, file):
        # add the {filename: file} entry to cache, record the insert order
        self._cache.update({filename: file})
        self._order.append(filename)
        # eliminate the cache if dict.size reached capacity
        if self._cache.__len__() > self._capacity:
            oldest_filename = self._order.pop(0)
            del self._cache[oldest_filename]


class IntersectDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_source,
            split,
            cache_capacity,
    ):
        self.data_source = data_source
        self.IBSfiles, self.pcd1files, self.pcd2files, self.pcd1gtfiles, self.pcd2gtfiles = \
            get_instance_filenames(data_source, split)

        self.cache_capacity = cache_capacity
        self.IBS_cache = DataCache(cache_capacity)
        self.pcd1gt_cache = DataCache(cache_capacity)
        self.pcd2gt_cache = DataCache(cache_capacity)

        logging.debug(
            "using "
            + str(len(self.pcd1files))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.pcd1files)

    def __getitem__(self, idx):
        IBS_filename = os.path.join(
            self.data_source, ws.IBS_gt_subdir, self.IBSfiles[idx]
        )
        pcd1_filename = os.path.join(
            self.data_source, ws.pcd_partial_subdir, self.pcd1files[idx]
        )
        pcd2_filename = os.path.join(
            self.data_source, ws.pcd_partial_subdir, self.pcd2files[idx]
        )
        pcd1gt_filename = os.path.join(
            self.data_source, ws.pcd_complete_subdir, self.pcd1gtfiles[idx]
        )
        pcd2gt_filename = os.path.join(
            self.data_source, ws.pcd_complete_subdir, self.pcd2gtfiles[idx]
        )

        IBS = get_pcd_data(IBS_filename)
        pcd1 = get_pcd_data(pcd1_filename)
        pcd2 = get_pcd_data(pcd2_filename)
        pcd1gt = get_pcd_data(pcd1gt_filename)
        pcd2gt = get_pcd_data(pcd2gt_filename)

        return IBS, pcd1, pcd2, pcd1gt, pcd2gt, idx
