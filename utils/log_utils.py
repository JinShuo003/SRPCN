"""
日志工具
"""
import logging
import os.path

from utils import path_utils


def get_logger(log_dir: str, scene: str, file_handler_level=logging.INFO, stream_handler_level=logging.INFO):
    _logger = logging.getLogger()
    _logger.setLevel("DEBUG")

    path_utils.generate_path(log_dir)
    log_filename = "{}.log".format(scene)
    log_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_path.format(scene), mode="w")
    file_handler.setLevel(level=file_handler_level)
    _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=stream_handler_level)
    _logger.addHandler(stream_handler)

    return _logger, file_handler, stream_handler



class LogFactory():
    created_loggers = dict()

    @staticmethod
    def get_logger(log_options: dict):
        log_tag = log_options.get("TAG")
        if log_tag not in LogFactory.created_loggers.keys():
            LogFactory.create_logger(log_options)

        return LogFactory.created_loggers[log_tag]


    @staticmethod
    def create_logger(logger_options: dict):
        log_tag = logger_options.get("TAG")
        log_type = logger_options.get("Type")
        log_dir = logger_options.get("LogDir")
        log_level = logger_options.get("GlobalLevel")
        log_file_level = logger_options.get("FileLevel")
        log_stream_level = logger_options.get("StreamLevel")
        log_mode = logger_options.get("Mode")

        logger = logging.getLogger()
        logger.setLevel(log_level)
        formatter = logging.Formatter("{} - %(levelname)s - %(message)s".format(log_tag))

        logger_path = os.path.join(log_dir, log_type, log_tag)
        if not os.path.isdir(logger_path):
            os.makedirs(logger_path)
        logger_path = os.path.join(logger_path, "logs.log")

        file_handler = None
        file_handler = logging.FileHandler(logger_path, mode=log_mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level=log_file_level)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level=log_stream_level)
        logger.addHandler(stream_handler)

        LogFactory.created_loggers[log_tag] = logger
