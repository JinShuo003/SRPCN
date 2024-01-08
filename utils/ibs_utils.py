import logging
import random
import time

import libibs
import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh

from utils import geometry_utils
from utils.geometry_utils import trimesh2o3d, get_pcd_from_np


class Log:
    def __init__(self, logger, text=""):
        self.logger = logger
        self.text = text

    def __enter__(self):
        self.begin_time = time.time()
        if self.logger is not None:
            self.logger.info("begin {}".format(self.text))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.logger is not None:
            self.logger.info("end {}, coast time: {}".format(self.text, self.end_time - self.begin_time))
            self.logger.info("")


class IBS:
    def __init__(self, trimesh_obj1: trimesh.Trimesh, trimesh_obj2: trimesh.Trimesh,
                 subdivide_max_edge: float = 0.05, sample_method: str = "poisson_disk", sample_num: int = 2048,
                 clip_border_type: str = "sphere", clip_sphere_radius: float = 1, clip_border_magnification: float = 1,
                 max_iterate_time: float = 10, show_iterate_result: bool = False, max_resample_points: int = 25000,
                 max_points_for_compute: int = 50000, simplify: bool = False, max_triangle_num: int = 50000,
                 logger: logging.Logger = None):
        """
        Args:
            trimesh_obj1: object1: trimesh.Trimesh
            trimesh_obj2: object2: trimesh.Trimesh
            subdivide_max_edge: The max length of triangle edge after subdivide
            sample_method: The method of sample points from mesh
            sample_num: The number of sample points from mesh to compute Voronoi Diagram
            clip_border_type: The type of border to clip ibs
            clip_sphere_radius: Make sense when $clip_border_type$ is "sphere", the radius of sphere
            clip_border_magnification: The magnification of clip border
            max_iterate_time: Maximum number of iterations
            show_iterate_result: If True, will show [mesh1, mesh2, ibs] after every iteration
            max_resample_points: The maximum number of resample points
            max_points_for_compute: The maximum number of points to compute Voronoi Diagram
            simplify: If True, will simplify ibs mesh until the number of triangles less than $max_triangle_num$
            max_triangle_num: Make sense when $simplify$ is True, the maximum number of triangles of ibs mesh
            logger: The logger to trace log
        """
        self.trimesh_obj1 = trimesh_obj1
        self.trimesh_obj2 = trimesh_obj2
        self.subdivide_max_edge = subdivide_max_edge
        self.sample_method = sample_method
        self.sample_num = sample_num
        self.clip_border_type = clip_border_type
        self.clip_sphere_radius = clip_sphere_radius
        self.clip_border_magnification = clip_border_magnification
        self.max_iterate_time = max_iterate_time
        self.show_iterate_result = show_iterate_result
        self.max_resample_points = max_resample_points
        self.max_points_for_compute = max_points_for_compute
        self.simplify = simplify
        self.max_triangle_num = max_triangle_num
        self.logger = logger if logger is not None else self.get_logger()
        self.ibs = None
        self.o3d_obj1 = trimesh2o3d(self.trimesh_obj1)
        self.o3d_obj2 = trimesh2o3d(self.trimesh_obj2)

        self.init_points1 = None  # init sample points from mesh1
        self.init_points2 = None  # init sample points from mesh2
        self.points1 = None  # points on mesh1 to compute Voronoi Diagram
        self.points2 = None  # points on mesh2 to compute Voronoi Diagram
        self.border_sphere_center = None  # The center of clip sphere
        self.border_sphere_radius = None  # The radius of clip sphere

    def launch(self):
        with Log(self.logger, "subdivide mesh1"):
            self.trimesh_obj1 = self._subdivide_mesh(self.trimesh_obj1, self.subdivide_max_edge)
            self._log_info("obj1 has {} faces after subdivide".format(self.trimesh_obj1.faces.shape[0]))

        with Log(self.logger, "subdivide mesh2"):
            self.trimesh_obj2 = self._subdivide_mesh(self.trimesh_obj2, self.subdivide_max_edge)
            self._log_info("obj2 has {} faces after subdivide".format(self.trimesh_obj2.faces.shape[0]))

        with (Log(self.logger, "get init sample points")):
            self.init_points1, self.init_points2 = \
                self._sample_points(self.trimesh_obj1, self.trimesh_obj2, self.sample_num)
            self.points1, self.points2 = self.init_points1, self.init_points2

        with Log(self.logger, "get clip border"):
            self.border_sphere_center, self.border_sphere_radius = self._get_clip_border()

        with Log(self.logger, "create ibs"):
            self._compute_ibs()

        with Log(self.logger, "cluster triangles"):
            self._remove_disconnected_mesh()

    def get_ibs_trimesh(self):
        return self.ibs

    def get_ibs_o3d(self):
        return geometry_utils.trimesh2o3d(self.ibs)

    def _get_logger(self):
        logger = logging.getLogger()
        logger.setLevel("INFO")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    def _log_info(self, msg: str):
        if self.logger is None:
            return
        self.logger.info(msg)

    def _get_clip_border(self):
        """
        Get clip sphere according to config info
        """
        if self.clip_border_type == "total":
            points = np.concatenate((self.points1, self.points2), axis=0)
            pcd = get_pcd_from_np(points)
            center, radius = geometry_utils.get_pcd_normalize_para(pcd)
        elif self.clip_border_type == "sphere":
            center = np.array([0, 0, 0])
            radius = self.clip_sphere_radius
        elif self.clip_border_type == "min_obj":
            pcd1 = get_pcd_from_np(self.points1)
            pcd2 = get_pcd_from_np(self.points2)
            center1, radius1 = geometry_utils.get_pcd_normalize_para(pcd1)
            center2, radius2 = geometry_utils.get_pcd_normalize_para(pcd2)
            center = center1 if radius1 < radius2 else center2
            radius = radius1 if radius1 < radius2 else radius2
        elif self.clip_border_type == "max_obj":
            pcd1 = get_pcd_from_np(self.points1)
            pcd2 = get_pcd_from_np(self.points2)
            center1, radius1 = geometry_utils.get_pcd_normalize_para(pcd1)
            center2, radius2 = geometry_utils.get_pcd_normalize_para(pcd2)
            center = center1 if radius1 > radius2 else center2
            radius = radius1 if radius1 > radius2 else radius2
        else:
            raise Exception("unsupported clip border type")
        return center, radius * self.clip_border_magnification

    def _subdivide_mesh(self, trimesh_obj: trimesh.Trimesh, max_edge_length: int):
        """
        subdivide the input mesh, until all edges of triangles less than max_edge_length
        Args:
            trimesh_obj: The mesh you want to subdivide
            max_edge_length: The maximum length of edges of triangles
        """
        vertices, faces = trimesh.remesh.subdivide_to_size(trimesh_obj.vertices, trimesh_obj.faces, max_edge_length)
        return trimesh.Trimesh(vertices, faces, process=True)

    def _compute_ibs_once(self):
        """
        Compute ibs according to self.points1 and self.points2
        """
        n0 = len(self.points1)
        n1 = len(self.points2)
        self._log_info("{} points on obj1, {} points on obj2".format(n0, n1))

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
        self.ibs.remove_degenerate_faces()

        if self.simplify and np.asarray(self.ibs.triangles).shape[0] > self.max_triangle_num:
            self._log_info("{} faces in ibs, need to be simplified".format(self.ibs.triangles.shape[0]))
            ibs_simplified = geometry_utils.trimesh2o3d(self.ibs).simplify_quadric_decimation(self.max_triangle_num)
            self.ibs = geometry_utils.o3d2trimesh(ibs_simplified)
            self._log_info("{} faces in ibs after simplify".format(self.ibs.triangles.shape[0]))

        if self.show_iterate_result:
            ibs_o3d = trimesh2o3d(self.ibs)
            ibs_o3d.paint_uniform_color((1, 0, 0))
            self.o3d_obj1.paint_uniform_color((0, 1, 0))
            self.o3d_obj2.paint_uniform_color((0, 0, 1))
            self._visualize([ibs_o3d, self.o3d_obj1, self.o3d_obj2])

    def _compute_ibs(self):
        """
        Compute ibs, ensure no intersection
        """
        collision_tester = trimesh.collision.CollisionManager()
        collision_tester.add_object('obj1', self.trimesh_obj1)
        collision_tester.add_object('obj2', self.trimesh_obj2)
        is_collide = True

        cur_iteration_num = 0

        contact_points_obj1 = []
        contact_points_obj2 = []
        while is_collide and cur_iteration_num < self.max_iterate_time:
            self._log_info("\niterate {}".format(cur_iteration_num))

            contact_points_obj1 = []
            contact_points_obj2 = []

            self._compute_ibs_once()

            is_collide, collision_data = collision_tester.in_collision_single(self.ibs, return_data=True)
            if not is_collide:
                break

            # get contact points on mesh1 and mesh2
            for i in range(len(collision_data)):
                if "obj1" in collision_data[i].names:
                    contact_points_obj1.append(collision_data[i].point)
                if "obj2" in collision_data[i].names:
                    contact_points_obj2.append(collision_data[i].point)

            contact_points_obj1_num = len(contact_points_obj1)
            contact_points_obj2_num = len(contact_points_obj2)
            contact_points_obj1 = np.array(contact_points_obj1)
            contact_points_obj2 = np.array(contact_points_obj2)

            # if collision occured, resample points near collision area and update points which are used to compute ibs
            if contact_points_obj1_num > 0:
                self._log_info("collision occured in obj1, size: {}".format(contact_points_obj1_num))
                points = self._resample_points(self.trimesh_obj1, contact_points_obj1, contact_points_obj2)
                self.points1 = np.concatenate((self.points1, points), axis=0)
                if self.points1.shape[0] > self.max_points_for_compute:
                    self.points1 = np.asarray(
                        geometry_utils.get_pcd_from_np(self.points1).farthest_point_down_sample(
                            self.max_points_for_compute).points)
                self.points1 = np.concatenate((self.init_points1, self.points1), axis=0)
                self.points1 = np.unique(self.points1, axis=0)

            if contact_points_obj2_num > 0:
                self._log_info("collision occured in obj2, size: {}".format(contact_points_obj2_num))
                points = self._resample_points(self.trimesh_obj2, contact_points_obj2, contact_points_obj1)
                self.points2 = np.concatenate((self.points2, points), axis=0)
                if self.points2.shape[0] > self.max_points_for_compute:
                    self.points2 = np.asarray(
                        geometry_utils.get_pcd_from_np(self.points2).farthest_point_down_sample(
                            self.max_points_for_compute).points)
                self.points2 = np.concatenate((self.init_points2, self.points2), axis=0)
                self.points2 = np.unique(self.points2, axis=0)

            cur_iteration_num += 1

        if cur_iteration_num == self.max_iterate_time:
            self._log_info("create ibs failed after {} iterates, concat with obj1: {}, obj2: {}".
                           format(self.max_iterate_time, len(contact_points_obj1), len(contact_points_obj2)))

    def _remove_disconnected_mesh(self):
        """
        Remove disconnected mesh, remain the main ibs mesh
        """
        ibs_o3d = trimesh2o3d(self.ibs)
        cluster_idx, triangle_num, area = ibs_o3d.cluster_connected_triangles()
        cluster_idx = np.asarray(cluster_idx)
        max_cluster_idx = triangle_num.index(max(triangle_num))
        disconnected_triangle_idx = []
        for i in range(cluster_idx.shape[0]):
            if cluster_idx[i] != max_cluster_idx:
                disconnected_triangle_idx.append(i)
        ibs_o3d.remove_triangles_by_index(disconnected_triangle_idx)
        ibs_o3d.remove_unreferenced_vertices()

        self.ibs = geometry_utils.o3d2trimesh(ibs_o3d)

    def _resample_points(self, mesh: trimesh.Trimesh, contact_points_obj1: np.ndarray, contact_points_obj2: np.ndarray):
        """
        Resample points on mesh near collision area
        """
        resample_points_clip_mesh = self._resample_points_on_clip_mesh(mesh, contact_points_obj1)
        resample_points_projection = self._resample_with_projection(mesh, contact_points_obj2)
        resample_points = np.concatenate((contact_points_obj1,
                                          resample_points_clip_mesh,
                                          resample_points_projection), axis=0)
        self._log_info("resample points num: {}".format(resample_points.shape[0]))
        if resample_points.shape[0] > self.max_resample_points:
            resample_points = np.asarray(
                geometry_utils.get_pcd_from_np(resample_points).farthest_point_down_sample(
                    self.max_resample_points).points)
            self._log_info("reduce resample points to {}".format(self.max_resample_points))

        return resample_points

    def _resample_points_on_clip_mesh(self, mesh: trimesh.Trimesh, contact_points_obj: np.ndarray):
        """
        Clip mesh according to collision area and sample points on mesh clipped
        """
        pcd_contact_obj = get_pcd_from_np(np.array(contact_points_obj))
        centroid, radius = geometry_utils.get_pcd_normalize_para(pcd_contact_obj)
        sphere = pv.Sphere(radius, centroid)
        pv_obj = geometry_utils.trimesh2pyvista(mesh)
        mesh_clip = geometry_utils.pyvista2o3d(pv_obj.clip_surface(sphere, invert=True))
        if np.asarray(mesh_clip.triangles).shape[0] == 0:
            return np.array([]).reshape(-1, 3)
        points = np.asarray(mesh_clip.sample_points_poisson_disk(128).points)
        return np.unique(points, axis=0)

    def _resample_with_projection(self, mesh: trimesh.Trimesh, contact_points_obj2: np.ndarray):
        """
        Project contact points to mesh
        """
        if contact_points_obj2.shape[0] == 0:
            return np.array([]).reshape(-1, 3)
        (nearest_points, __, __) = mesh.nearest.on_surface(contact_points_obj2)
        return np.unique(nearest_points, axis=0)

    def _sample_points(self, trimesh_obj1: trimesh.Trimesh, trimesh_obj2: trimesh.Trimesh, points_num: int):
        """
        Sample points on mesh1 and mesh2
        """
        if self.sample_method == "poisson_disk":
            return (self._sample_points_poisson_disk(trimesh_obj1, points_num),
                    self._sample_points_poisson_disk(trimesh_obj2, points_num))
        elif self.sample_method == "dist_weight":
            return self._sample_points_with_dist_weight(trimesh_obj1, trimesh_obj2, points_num)

    def _sample_points_poisson_disk(self, trimesh_obj, points_num):
        """
        Sample points on mesh, method: poisson disk
        """
        o3d_mesh = trimesh2o3d(trimesh_obj)
        pcd = o3d_mesh.sample_points_poisson_disk(points_num)
        return np.asarray(pcd.points)

    def _sample_points_with_dist_weight(self, trimesh_obj1: trimesh.Trimesh, trimesh_obj2: trimesh.Trimesh, points_num):
        """
        Sample points on mesh, the weights of triangles are positively correlated with distance
        between triangle with another mesh.
        """
        init_points_num = int(1.5 * points_num)
        o3d_obj1 = trimesh2o3d(trimesh_obj1)
        o3d_obj2 = trimesh2o3d(trimesh_obj2)
        sample_points1 = np.asarray(o3d_obj1.sample_points_poisson_disk(init_points_num).points)
        sample_points2 = np.asarray(o3d_obj2.sample_points_poisson_disk(init_points_num).points)

        weights1 = 1 / abs(trimesh.proximity.signed_distance(trimesh_obj2, sample_points1))
        weights2 = 1 / abs(trimesh.proximity.signed_distance(trimesh_obj1, sample_points2))
        weights1[np.isinf(weights1)] = 100
        weights2[np.isinf(weights2)] = 100
        weights1 /= sum(weights1)
        weights2 /= sum(weights2)

        sample_points1_idx = np.random.choice(range(init_points_num), points_num, False, weights1)
        sample_points2_idx = np.random.choice(range(init_points_num), points_num, False, weights2)
        sample_points1 = sample_points1[sample_points1_idx]
        sample_points2 = sample_points2[sample_points2_idx]

        return sample_points1, sample_points2

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
