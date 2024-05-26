"""
自定义异常
"""


class SampleMethodException(Exception):
    def __init__(self, message="Illegal sample method, surface or IOU are supported"):
        self.message = message
        super.__init__(message)


class BorderNotSetException(Exception):
    def __init__(self):
        self.msg = "Border haven't been set"


class DirectionNotSetException(Exception):
    def __init__(self):
        self.msg = "Direction is None, you should set border first, the direction will be computed automatically"


class DataTypeInvalidException(Exception):
    def __init__(self, type: str):
        self.msg = "The type of the point should be {}".format(type)


class DataDemensionInvalidException(Exception):
    def __init__(self, dimension):
        self.msg = "The demension of data should be {}".format(dimension)