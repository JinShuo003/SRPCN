import open3d as o3d
from utils import geometry_utils
import numpy as np


sphere = geometry_utils.get_sphere_pcd(radius=0.5)
coordinate = geometry_utils.get_coordinate(size=0.1)
arrow = geometry_utils.get_arrow(np.array([1, 0, 0]), np.array([0.5, 0, 0], dtype=np.float32))
o3d.visualization.draw_geometries([arrow, sphere, coordinate])

