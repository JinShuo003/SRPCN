import logging
import os.path

from utils import path_utils


def get_logger(log_dir: str, scene: str):
    _logger = logging.getLogger()
    _logger.setLevel("INFO")

    path_utils.generate_path(log_dir)
    log_filename = "{}.log".format(scene)
    log_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_path.format(scene), mode="w")
    file_handler.setLevel(level=logging.INFO)
    _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    _logger.addHandler(stream_handler)

    return _logger, file_handler, stream_handler
