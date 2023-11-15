import json
import os

pcd_partial_subdir = "pcdScan"
pcd_complete_subdir = "pcdComplete"
IBS_gt_subdir = "IBS"
scene_patten = "scene\\d\\.\\d{4}"


def load_experiment_specifications(experiment_config_file):

    if not os.path.isfile(experiment_config_file):
        raise Exception("The experiment config file does not exist")

    return json.load(open(experiment_config_file))
