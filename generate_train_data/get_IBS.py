"""
多进程计算点云的ibs面
"""
import logging
import multiprocessing
import os
import re

import open3d as o3d

from utils import geometry_utils, path_utils, ibs_utils, log_utils


def save_ibs_mesh(specs, scene, ibs_mesh_o3d):
    mesh_dir = specs['ibs_mesh_save_dir']
    category = re.match(specs['category_re'], scene).group()
    # mesh_dir+category不存在则创建目录
    if not os.path.isdir(os.path.join(mesh_dir, category)):
        os.makedirs(os.path.join(mesh_dir, category))

    ibs_mesh_filename = '{}.obj'.format(scene)
    mesh_path = os.path.join(mesh_dir, category, ibs_mesh_filename)
    o3d.io.write_triangle_mesh(mesh_path, ibs_mesh_o3d)


class TrainDataGenerator:
    def __init__(self, specs, logger):
        self.specs = specs
        self.logger = logger

    def get_ibs_mesh_o3d(self, geometries_path: dict):
        mesh1 = geometry_utils.read_mesh(geometries_path["mesh1"])
        mesh2 = geometry_utils.read_mesh(geometries_path["mesh2"])

        mesh1_trimesh = geometry_utils.o3d2trimesh(mesh1)
        mesh2_trimesh = geometry_utils.o3d2trimesh(mesh2)

        ibs = ibs_utils.IBSMesh(init_size_sampling=256,
                                resamplings=5,
                                improve_by_collision=True)
        ibs.execute(mesh1_trimesh, mesh2_trimesh)
        ibs_mesh_o3d = geometry_utils.trimesh2o3d(ibs.get_trimesh())

        mesh1.paint_uniform_color((1, 0, 0))
        mesh2.paint_uniform_color((0, 1, 0))
        ibs_mesh_o3d.paint_uniform_color((0, 0, 1))

        o3d.visualization.draw_geometries([mesh1, mesh2, ibs_mesh_o3d], mesh_show_wireframe=True)

        return ibs_mesh_o3d

    def handle_scene(self, scene):
        """处理当前场景，包括采集多角度的残缺点云、计算直接法和间接法网络的sdf gt、计算残缺点云下的ibs"""
        # ------------------------------获取点云数据，包括完整点云和各个视角的残缺点云--------------------------
        geometries_path = path_utils.get_geometries_path(self.specs, scene)
        ibs_mesh_o3d = self.get_ibs_mesh_o3d(geometries_path)

        save_ibs_mesh(self.specs, scene, ibs_mesh_o3d)


def my_process(scene, specs):
    _logger = logging.getLogger()
    _logger.setLevel("INFO")
    file_handler = log_utils.add_file_handler(_logger, "logs/get_IBS", f"{scene}.log")

    process_name = multiprocessing.current_process().name
    _logger.info(f"Running task in process: {process_name}, scene: {scene}")
    trainDataGenerator = TrainDataGenerator(specs, _logger)

    try:
        trainDataGenerator.handle_scene(scene)
        _logger.info("scene: {} succeed".format(scene))
    except Exception as e:
        _logger.error("scene: {} failed, exception message: {}".format(scene, e.message))
    finally:
        log_utils.remove_file_handler(_logger, file_handler)


if __name__ == '__main__':
    config_filepath = 'configs/get_IBS.json'
    specs = path_utils.read_config(config_filepath)
    filename_tree = path_utils.get_filename_tree(specs, specs.get("path_options").get("geometries_dir").get("mesh_dir"))
    path_utils.generate_path(specs.get("path_options").get("ibs_mesh_save_dir"))

    pool = multiprocessing.Pool(processes=8)

    # 参数
    view_list = []
    for category in filename_tree:
        for scene in filename_tree[category]:
            for filename in filename_tree[category][scene]:
                view_list.append(filename)

    # for filename in view_list:
    #     category_num = int(filename[5])-1
    #     pool.apply_async(my_process, (filename, specs))
    #
    # # 关闭进程池
    # pool.close()
    # pool.join()

    trainDataGenerator = TrainDataGenerator(specs, None)
    for filename in view_list:
        category_num = int(filename[5])-1
        trainDataGenerator.handle_scene(filename)

