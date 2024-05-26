"""
随机数工具
"""
import random

import numpy as np
import math


def randNormalFloat(l: float, h: float, num: int):
    """生成num个给定范围内的浮点数，符合正态分布"""
    if l > h:
        return None
    else:
        return np.random.normal((h + l) / 2, (h - l) / 6, num)


def randUniFormFloat(l: float, h: float, num: int):
    """生成num个给定范围内的浮点数，符合均匀分布"""
    if l > h:
        return None
    else:
        return np.array([np.random.random() * (h - l) + l for i in range(num)])


def randPointsUniform(num: int, radius: float):
    u = np.random.uniform(size=(num, 1))
    v = np.random.uniform(size=(num, 1))
    theta = u * 2.0 * math.pi
    phi = np.arccos(2.0 * v - 1.0)
    r = np.cbrt(np.random.uniform(size=(num, 1))) * radius
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    x = r * sinPhi * cosTheta
    y = r * sinPhi * sinTheta
    z = r * cosPhi
    return np.concatenate([x, y, z], axis=1)


def random_offset(point, d: float):
    """将point向随机方向偏移最大为d的随机距离"""
    point = np.array(point)
    # 生成随机方向向量
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)  # 归一化为单位向量

    # 生成随机长度
    length = np.random.uniform(0, d)

    # 计算偏移点的坐标
    offset = point + direction * length

    return offset


def get_random_points_in_aabb(aabb, points_num):
    """在aabb范围内以均匀的方式采集points_num个点"""
    # 获取包围框边界点
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    random_points = []
    for i in range(3):
        random_points.append(
            randUniFormFloat(min_bound[i], max_bound[i], int(points_num)).reshape((-1, 1)))
    random_points = np.concatenate([random_points[0], random_points[1], random_points[2]], axis=1)

    return random_points


def get_random_points_in_sphere(num_points, center=[0, 0, 0], radius=0.5):
    """
    在球心为center，半径为radius的球内均匀采集num_points个点
    :param center: 球心
    :param radius: 半径
    :param num_points: 点数
    :return: list(np.ndarray)
    """
    points = []
    for _ in range(num_points):
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        w = np.random.uniform(0, 1)

        r = radius * (u ** (1 / 3))
        theta = 2 * np.pi * v
        phi = np.arccos(2 * w - 1)

        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)

        points.append([x, y, z])

    return points


def get_random_points_from_seeds(seeds, rate, radius):
    """从种子点出发，在球形范围内随机散点，方向向量和距离均采用随机值"""
    random_points = []
    for seed in seeds:
        cur_random_points = get_random_points_in_sphere(rate, seed, radius)
        random_points += cur_random_points
    return random_points


def get_random_points_with_limit(aabb, points_num, radius=0.5):
    """在aabb范围内以均匀的方式采集points_num个点，截断超出半径为radius的球的点"""
    # 获取包围框边界点
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    random_points = []
    while len(random_points) < points_num:
        cur_random_points = []
        for i in range(3):
            cur_random_points.append(
                randUniFormFloat(min_bound[i], max_bound[i], int(points_num * 0.6)).reshape((-1, 1)))
        cur_random_points = np.concatenate([cur_random_points[0], cur_random_points[1], cur_random_points[2]], axis=1)
        cur_random_points = [point for point in cur_random_points if np.linalg.norm(point) <= radius]
        random_points.extend(cur_random_points)
    random_points = random.sample(random_points, int(points_num))

    return random_points
