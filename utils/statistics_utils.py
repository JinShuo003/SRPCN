import csv
import re
import os
import pandas as pd

INTE_name_dict = {
    "scene1": "desk-chair",
    "scene2": "hook-bag",
    "scene3": "vace-flower",
    "scene4": "shelf-hanger",
    "scene5": "hanger-clothes",
    "scene6": "basket-object",
    "scene7": "shelf-object",
    "scene8": "cart-object",
    "scene9": "shelf-cap",
}

C3d_name_dict = {
    "scene1": "airplane",
    "scene2": "cabinet",
    "scene3": "car",
    "scene4": "chair",
    "scene5": "lamp",
    "scene6": "sofa",
    "scene7": "table",
    "scene8": "watercraft",
    "scene9": "bed",
    "scene10": "bench",
    "scene11": "bookself",
    "scene12": "bus",
    "scene13": "guitar",
    "scene14": "motorbike",
    "scene15": "pistol",
    "scene16": "skateboard",
}


INTE_indicators = [
    "cd_l1",
    "emd",
    "fscore",
    "mad_s",
    "mad_i",
    "ibs_a",
    "interact_num"
]

C3d_indicators = [
    "cd_l1",
    "cd_l2",
    "emd",
    "fscore"
]

indicator_scale = {
    "cd_l1": 1000,
    "emd": 10000,
    "fscore": 1,
    "mad_s": 1000,
    "mad_i": 1000000,
    "ibs_a": 100,
    "interact_num": 1
}


def write_avrg_csv_file(csv_path, json_data, dataset: str):
    if dataset == "C3d":
        name_dict = C3d_name_dict
        indicators = C3d_indicators
    elif dataset == "INTE":
        name_dict = INTE_name_dict
        indicators = INTE_indicators
    else:
        raise Exception("dataset not support")

    scene_name_patten = "scene\\d"
    scenes = sorted([key for key in json_data["cd_l1"].keys() if re.match(scene_name_patten, key)])
    scene_names = [name_dict[key] for key in scenes]
    index = [i for i in range(1, len(scenes) + 1)]

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([''] + index + ["avrg"])
        csv_writer.writerow([''] + scene_names)

        for indicator in indicators:
            avrg_dist_values = [json_data[indicator][scene]["avrg_dist"] * indicator_scale[indicator] for scene in
                                scenes]
            avrg_dist_values.append(json_data[indicator]["avrg_dist"] * indicator_scale[indicator])
            csv_writer.writerow([indicator] + avrg_dist_values)


def append_csv_data(csv_data: dict, filename_list: list, *args):
    assert len(args) > 0
    assert len(filename_list) == len(args[0])
    
    lists = [tensor.tolist() for tensor in args]
    for i, data in enumerate(zip(*lists)):
        filename = os.path.splitext(os.path.basename(filename_list[i]))[0]
        csv_data[filename] = list(data)
    

def write_single_csv_file(save_path: str, csv_data: dict):
    df = pd.DataFrame(csv_data)
    df.to_csv(save_path, index=False)
