import time


# 耗时统计
class CostTime(object):
    def __init__(self, action_describtion):
        self.t = 0
        self.action_describtion = action_describtion

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'{self.action_describtion} cost time:{time.perf_counter() - self.t:.8f} s')
