import open3d as o3d
import numpy as np

def buildO3dPcdFromNp(array):
    if array.ndim != 2 or array.shape[1] != 3:
        print("the dimension of the input is illegal")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)

    return pcd
