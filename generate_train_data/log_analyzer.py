import argparse
import os
import logging
import re


logger = None


def begin_analyze(log_dir, patten):
    for filename in os.listdir(log_dir):
        if not filename.split('.')[-1] == "log":
            continue
        with open(os.path.join(log_dir, filename), 'r') as f:
            line = f.readline()
            flag = False
            while line:
                if re.match(patten, line):
                    if not flag:
                        logger.info("")
                        logger.info(filename)
                    logger.info(line.split('\n')[0])
                    flag = True
                line = f.readline()


if __name__ == '__main__':
    log_dir = "logs/get_scan_pcd"
    patten = ".*not enough init points.*"

    arg_parser = argparse.ArgumentParser(description="Analyze log line by line")
    arg_parser.add_argument(
        "--log_dir",
        "-l",
        dest="log_dir",
        default=log_dir,
        required=False,
        help="The directory of log files"
    )
    arg_parser.add_argument(
        "--patten",
        "-p",
        dest="patten",
        default=patten,
        required=False,
        help="The patten of the line you interested"
    )
    args = arg_parser.parse_args()

    logger = logging.getLogger("log_analyzer")
    logger.setLevel("INFO")
    file_handler = logging.FileHandler("log_analyzer_result.txt", mode="w")
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    begin_analyze(args.log_dir, args.patten)
