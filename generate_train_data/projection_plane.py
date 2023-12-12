import numpy as np
from utils import exception_utils


class Plane:
    """
    空间平面，用于计算射线
    该平面只有两个旋转自由度，永远不会有滚转角，即矩形的上下边永远平行于x-y平面
    关键参数：矩形的四个边界点、四个方向向量
    """
    def __init__(self):
        # 平面的四个角点，左上、左下、右上、右下
        self.border: tuple = None
        # 平面的四个方向向量，左、右、上、下
        self.direction: tuple = None

    def set_border(self, border: tuple):
        required_border_type = tuple
        required_border_size = 4
        required_point_type = np.ndarray
        required_point_shape = 3,
        if not isinstance(border, required_border_type):
            raise exception_utils.DataTypeInvalidException(required_border_type)
        if not border.__len__() == required_border_size:
            raise exception_utils.DataDemensionInvalidException(required_border_size)
        for point in border:
            if not isinstance(point, required_point_type):
                raise exception_utils.DataTypeInvalidException()
            if not point.shape == required_point_shape:
                raise exception_utils.DataDemensionInvalidException(required_point_shape)
        self.border = border
        self._compute_direction()

    def get_border(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border

    def get_left_up(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[0]

    def get_left_down(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[1]

    def get_right_up(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[2]

    def get_right_down(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        return self.border[3]

    def _compute_direction(self):
        if self.border is None:
            raise exception_utils.BorderNotSetException()
        left_up = self.get_left_up()
        left_down = self.get_left_down()
        right_up = self.get_right_up()
        right_down = self.get_right_down()
        dir_left = (left_up - right_up) / np.linalg.norm(left_up - right_up)
        dir_right = (right_up - left_up) / np.linalg.norm(right_up - left_up)
        dir_up = (left_up - left_down) / np.linalg.norm(left_up - left_down)
        dir_down = (left_down - left_up) / np.linalg.norm(left_down - left_up)
        self.direction = (dir_left, dir_right, dir_up, dir_down)

    def get_dir_left(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[0]

    def get_dir_right(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[1]

    def get_dir_up(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[2]

    def get_dir_down(self):
        if self.direction is None:
            raise exception_utils.DirectionNotSetException()
        return self.direction[3]
