import random

import libibs
import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh

from utils import geometry_utils
from utils.geometry_utils import trimesh2o3d, get_pcd_from_np


class IBS:
    def __init__(self, trimesh_obj1, trimesh_obj2, logger=None):
        self.logger = logger

        self.log_info("begin subdevide mesh1")
        self.trimesh_obj1 = self._subdevide_mesh(trimesh_obj1, 0.05)
        self.log_info("end subdevide mesh1")

        self.log_info("begin subdevide mesh2")
        self.trimesh_obj2 = self._subdevide_mesh(trimesh_obj2, 0.05)
        self.log_info("end subdevide mesh2")

        self.o3d_obj1 = trimesh2o3d(self.trimesh_obj1)
        self.o3d_obj2 = trimesh2o3d(self.trimesh_obj2)

        self.log_info("begin sample points")
        self.points1 = self._sample_points_poisson_disk(self.trimesh_obj1, 2048)
        self.points2 = self._sample_points_poisson_disk(self.trimesh_obj2, 2048)
        self.log_info("end sample points")

        self.log_info("begin get clip border")
        self.border_sphere_center, self.border_sphere_radius = self._get_clip_border()
        self.border_sphere_radius *= 2
        self.log_info("end get clip border")

        self.ibs = None
        self.log_info("create_ibs_mesh begin")
        self._create_ibs_mesh()
        self.log_info("create_ibs_mesh end")

    def log_info(self, msg: str):
        if self.logger is None:
            return
        self.logger.info(msg)

    def _get_clip_border(self):
        pcd1 = get_pcd_from_np(self.points1)
        pcd2 = get_pcd_from_np(self.points2)
        center1, radius1 = geometry_utils.get_pcd_normalize_para(pcd1)
        center2, radius2 = geometry_utils.get_pcd_normalize_para(pcd2)
        if radius1 < radius2:
            return center1, radius1
        else:
            return center2, radius2

    def _subdevide_mesh(self, trimesh_obj, max_edge):
        """
        将mesh进行细分，直到所有edge都小于max_edge
        """
        vertices, faces = trimesh.remesh.subdivide_to_size(trimesh_obj.vertices, trimesh_obj.faces, max_edge)
        return trimesh.Trimesh(vertices, faces, process=True)

    def _get_current_ibs(self):
        """
        计算ibs，求出两物体外接球中较小的那个，作为截断边界
        """
        n0 = len(self.points1)
        n1 = len(self.points2)

        n2 = (n0 + n1) // 10
        shell = fibonacci_sphere(n2)
        shell = shell * self.border_sphere_radius + self.border_sphere_center

        points = np.concatenate([
            self.points1,
            self.points2]).astype('float32')

        points = np.concatenate([points, shell])
        ids = np.zeros(n0 + n1 + n2).astype('int32')
        ids[n0:] = 1
        ids[n0 + n1:] = 2

        v, f, p = libibs.create_ibs(np.concatenate([points]), ids)
        f = f[~(p >= n0 + n1).any(axis=-1)]

        ibs = pv.make_tri_mesh(v, f)

        self.ibs = trimesh.Trimesh(ibs.points, ibs.faces.reshape(-1, 4)[:, 1:], process=False)
        self.ibs.remove_unreferenced_vertices()

    def _create_ibs_mesh(self):
        collision_tester = trimesh.collision.CollisionManager()
        collision_tester.add_object('obj1', self.trimesh_obj1)
        collision_tester.add_object('obj2', self.trimesh_obj2)
        in_collision = True

        max_iterate_time = 10
        iterate_time = 0

        contact_points_obj1 = []
        contact_points_obj2 = []
        while in_collision and iterate_time < max_iterate_time:
            contact_points_obj1 = []
            contact_points_obj2 = []

            self._get_current_ibs()
            in_collision, data = collision_tester.in_collision_single(self.ibs, return_data=True)
            if not in_collision:
                break

            self.log_info("iterate {}".format(iterate_time))

            for i in range(len(data)):
                if "obj1" in data[i].names:
                    contact_points_obj1.append(data[i].point)
                if "obj2" in data[i].names:
                    contact_points_obj2.append(data[i].point)

            if len(contact_points_obj1) > 0 or len(contact_points_obj2) > 0:
                iterate_time += 1

            np_contact_points_obj1 = np.array(contact_points_obj1)
            np_contact_points_obj2 = np.array(contact_points_obj2)

            if len(contact_points_obj1) > 0:
                self.log_info("collision occured in obj1, size: {}".format(len(contact_points_obj1)))
                points = self._resample_points(self.trimesh_obj1, np_contact_points_obj1, np_contact_points_obj2)
                if points.shape[0] != 0:
                    self.points1 = np.concatenate((self.points1, points), axis=0)

            if len(contact_points_obj2) > 0:
                self.log_info("collision occured in obj2, size: {}".format(len(contact_points_obj2)))
                points = self._resample_points(self.trimesh_obj2, np_contact_points_obj2, np_contact_points_obj1)
                self.points2 = np.concatenate((self.points2, points), axis=0)

        if iterate_time == max_iterate_time:
            self.log_info("create ibs failed after {} iterates, concat with obj1: {}, obj2: {}".
                          format(max_iterate_time, len(contact_points_obj1), len(contact_points_obj2)))

    def _resample_points(self, trimesh_obj1, contact_points_obj1, contact_points_obj2):
        """
        在obj1上进行再次散点
        """
        # 在截断mesh上散点
        resample_points_clip_mesh = self._resample_points_on_clip_mesh(trimesh_obj1, contact_points_obj1)
        # 将obj2的碰撞点投影到obj1的mesh上
        resample_points_projection = self._resample_with_projection(trimesh_obj1, contact_points_obj2)
        return np.concatenate((resample_points_clip_mesh, resample_points_projection), axis=0)

    def _resample_points_on_clip_mesh(self, trimesh_obj, contact_points_obj):
        """
        根据碰撞情况截取mesh，并在截取后的模型上散点
        """
        pcd_contact_obj = get_pcd_from_np(np.array(contact_points_obj))
        centroid, radius = geometry_utils.get_pcd_normalize_para(pcd_contact_obj)
        sphere = pv.Sphere(radius, centroid)
        pv_obj = geometry_utils.trimesh2pyvista(trimesh_obj)
        mesh_clip = geometry_utils.pyvista2o3d(pv_obj.clip_surface(sphere, invert=True))
        if np.asarray(mesh_clip.triangles).shape[0] == 0:
            return np.array([]).reshape(-1, 3)
        points = np.asarray(mesh_clip.sample_points_poisson_disk(128).points)
        return np.unique(points, axis=0)

    def _resample_with_projection(self, trimesh_obj1, contact_points_obj2):
        """
        将obj2的碰撞点投影到obj1
        """
        if contact_points_obj2.shape[0] == 0:
            return np.array([]).reshape(-1, 3)
        (nearest_points, __, __) = trimesh_obj1.nearest.on_surface(contact_points_obj2)
        return np.unique(nearest_points, axis=0)

    def _sample_points_in_sphere(self, o3d_mesh, centroid, radius, points_num):
        points = []
        sphere = geometry_utils.get_sphere_pcd(centroid, radius)
        sphere.paint_uniform_color((1, 0, 0))
        while len(points) < points_num:
            random_points = np.asarray(o3d_mesh.sample_points_uniformly(5*points_num).points)
            pcd = get_pcd_from_np(random_points)
            pcd.paint_uniform_color((0, 1, 0))
            self._visualize([pcd, sphere])
            distances = np.linalg.norm(random_points - centroid, axis=1)
            indices_inside = np.where(distances <= radius)[0]
            points_inside = random_points[indices_inside]
            pcd_inside = get_pcd_from_np(points_inside)
            pcd_inside.paint_uniform_color((0, 1, 0))
            self._visualize([pcd_inside, sphere])
            points += points_inside.tolist()

        points = np.array(random.sample(points, points_num))
        return points

    def _sample_points_with_dist_weight(self, trimesh_obj1: trimesh.Trimesh, trimesh_obj2: trimesh.Trimesh,
                                        points_num: int):
        """
        从Mesh中进行带权的点云采样，距离另一个物体越近权重越大
        """
        init_points_num = 2*points_num
        o3d_obj1 = trimesh2o3d(trimesh_obj1)
        o3d_obj2 = trimesh2o3d(trimesh_obj2)
        sample_points1 = np.asarray(o3d_obj1.sample_points_poisson_disk(init_points_num).points)
        sample_points2 = np.asarray(o3d_obj2.sample_points_poisson_disk(init_points_num).points)

        weights1 = 1 / abs(trimesh.proximity.signed_distance(trimesh_obj2, sample_points1))
        weights2 = 1 / abs(trimesh.proximity.signed_distance(trimesh_obj1, sample_points2))
        weights1 /= sum(weights1)
        weights2 /= sum(weights2)

        sample_points1_idx = np.random.choice(range(init_points_num), points_num, False, weights1)
        sample_points2_idx = np.random.choice(range(init_points_num), points_num, False, weights2)
        sample_points1 = sample_points1[sample_points1_idx]
        sample_points2 = sample_points2[sample_points2_idx]

        return sample_points1, sample_points2

    def _sample_points_poisson_disk(self, trimesh_obj, points_num):
        o3d_mesh = trimesh2o3d(trimesh_obj)
        pcd = o3d_mesh.sample_points_poisson_disk(points_num)
        return np.asarray(pcd.points)

    def _visualize(self, geometries: list):
        o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True, mesh_show_back_face=True)


def fibonacci_sphere(n=48, offset=False):
    """Sample points on sphere using fibonacci spiral.

    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    :param int n: number of sample points, defaults to 48
    :param bool offset: set True to get more uniform samplings when n is large , defaults to False
    :return array: points samples
    """

    golden_ratio = (1 + 5 ** 0.5) / 2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / golden_ratio

    if offset:
        if n >= 600000:
            epsilon = 214
        elif n >= 400000:
            epsilon = 75
        elif n >= 11000:
            epsilon = 27
        elif n >= 890:
            epsilon = 10
        elif n >= 177:
            epsilon = 3.33
        elif n >= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        phi = np.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))
    else:
        phi = np.arccos(1 - 2 * (i + 0.5) / n)

    x = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1)
    return x
