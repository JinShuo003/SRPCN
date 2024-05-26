"""
解压Completion3d数据集，并将其组织形式调整为INTE数据集的形式
"""
import os.path
import open3d as o3d

import h5py
from utils import geometry_utils, path_utils

if __name__ == '__main__':
    file_path = "D:\\dataset\\MVP_Test_CP.h5"
    pcd_complete_save_dir = "D:\\dataset\\IBPCDC\\pcdComplete\\MVP"
    pcd_incomplete_save_dir = "D:\\dataset\\IBPCDC\\pcdScan\\MVP"

    file = h5py.File(file_path, 'r')
    idx = 0
    label_list = file.get("labels")
    pcd_complete_list = file.get("complete_pcds")
    pcd_incomplete_list = file.get("incomplete_pcds")

    pcd_complete_amount = pcd_complete_list.len()
    data_amount = label_list.len()
    file_tree = {
        "scene1": 200,
        "scene2": 200,
        "scene3": 200,
        "scene4": 200,
        "scene5": 200,
        "scene6": 200,
        "scene7": 200,
        "scene8": 200,
        "scene9": 100,
        "scene10": 100,
        "scene11": 100,
        "scene12": 100,
        "scene13": 100,
        "scene14": 100,
        "scene15": 100,
        "scene16": 100,
    }

    """
    共有16个类别，每个类别中的每一条数据有26个视角
    """
    for pcd_complete_idx in range(pcd_complete_amount):
        pcd_complete = geometry_utils.get_pcd_from_np(pcd_complete_list[pcd_complete_idx])

        label = label_list[pcd_complete_idx * 26]
        category = "scene{}".format(label + 1)
        if category not in file_tree:
            file_tree[category] = 0
        scene_idx = file_tree[category]
        path_utils.generate_path(os.path.join(pcd_complete_save_dir, category))

        pcd_complete_filename = "{}.{:04d}.ply".format(category, scene_idx)
        pcd_complete_path = os.path.join(pcd_complete_save_dir, category, pcd_complete_filename)
        o3d.io.write_point_cloud(pcd_complete_path, pcd_complete)

        for view_idx in range(26):
            data_idx = pcd_complete_idx * 26 + view_idx

            pcd_incomplete = geometry_utils.get_pcd_from_np(pcd_incomplete_list[data_idx])

            path_utils.generate_path(os.path.join(pcd_incomplete_save_dir, category))

            pcd_incomplete_filename = "{}.{:04d}_view{}.ply".format(category, scene_idx, view_idx)
            pcd_incomplete_path = os.path.join(pcd_incomplete_save_dir, category, pcd_incomplete_filename)
            o3d.io.write_point_cloud(pcd_incomplete_path, pcd_incomplete)

        file_tree[category] += 1
