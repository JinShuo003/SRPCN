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


def get_train_logger(specs):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    formatter = logging.Formatter("IBPCDC - %(levelname)s - %(message)s")

    log_file_path = os.path.join(specs.get("LogDir"), specs.get("TAG"))
    if not os.path.isdir(log_file_path):
        os.makedirs(log_file_path)

    continue_train = specs.get("TrainOptions").get("ContinueTrain")
    file_handler = None
    if continue_train:
        file_handler = logging.FileHandler(os.path.join(log_file_path, "logs.log"), mode="a")
    else:
        file_handler = logging.FileHandler(os.path.join(log_file_path, "logs.log"), mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def get_test_logger(specs):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    formatter = logging.Formatter("IBPCDC - %(levelname)s - %(message)s")

    log_file_path = os.path.join(specs.get("LogDir"), specs.get("TAG"))
    if not os.path.isdir(log_file_path):
        os.makedirs(log_file_path)

    file_handler = logging.FileHandler(os.path.join(log_file_path, "logs.log"), mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    return logger