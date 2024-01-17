import open3d as o3d
from utils import geometry_utils
import numpy as np


def get_grid(points_num: int = 2048, nb_primitives: int = 4):
    points_num = 2048
    nb_primitives = 4
    grain = int(np.sqrt(points_num / nb_primitives))
    grain = grain * 1.0
    n = ((grain + 1) * (grain + 1) * nb_primitives)
    if n < points_num:
        grain += 1
    npts = (grain + 1) * (grain + 1) * nb_primitives
    print(npts)

    # generate regular grid
    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain])

    grid = [vertices for i in range(0, nb_primitives)]
    print("grain", grain, 'number vertices', len(vertices) * nb_primitives)
    return grid


grid = AtlasNet_setup()
print(grid)
