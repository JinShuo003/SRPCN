import csv
import json
import re


MVP_name_dict = {
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

indicators = [
    "cd_l1",
    "cd_l2",
    "emd",
    "fscore"
]

indicator_scale = {
    "cd_l1": 1000,
    "cd_l2": 10000,
    "emd": 10000,
    "fscore": 1
}


def save_json_as_csv(csv_path, json_data, dataset: str):
    if dataset == "MVP":
        name_dict = MVP_name_dict
    elif dataset == "INTE":
        name_dict = INTE_name_dict
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
