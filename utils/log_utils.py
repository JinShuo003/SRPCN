import logging
import os.path


def get_logger(log_options: dict):
    logger = logging.getLogger(log_options.get("logger_name"))
    logger.setLevel(level=logging.getLevelName(log_options.get("logger_level")))

    if log_options.get("out_to_file"):
        file_log_options = log_options.get("file_log_options")
        file_handler = logging.FileHandler(file_log_options.get("file_path"), mode="w")
        file_handler.setLevel(level=file_log_options.get("log_level"))
        if file_log_options.get("is_need_formatter"):
            formatter = logging.Formatter(file_log_options.get("formatter"))
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_options.get("out_to_console"):
        console_log_options = log_options.get("console_log_options")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level=console_log_options.get("log_level"))
        if console_log_options.get("is_need_formatter"):
            formatter = logging.Formatter(file_log_options.get("formatter"))
            stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def add_file_handler(logger, log_path, log_file_name):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, log_file_name), mode="w")
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)
    return file_handler


def remove_file_handler(logger, file_handler):
    logger.removeHandler(file_handler)
