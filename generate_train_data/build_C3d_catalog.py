"""
构造Completion3d数据集的目录文件
"""
import json
import os.path

if __name__ == '__main__':
    data_dir = "D:\dataset\IBPCDC\pcdScan\C3d"
    train_data_num = {
        "scene1": 200,
        "scene2": 200,
        "scene3": 200,
        "scene4": 200,
        "scene5": 200,
        "scene6": 200,
        "scene7": 200,
        "scene8": 200,
        "scene9": 100,
        "scene10": 100,
        "scene11": 100,
        "scene12": 100,
        "scene13": 100,
        "scene14": 100,
        "scene15": 100,
        "scene16": 100
    }
    test_data_num = {
        "scene1": 150,
        "scene2": 150,
        "scene3": 150,
        "scene4": 150,
        "scene5": 150,
        "scene6": 150,
        "scene7": 150,
        "scene8": 150,
        "scene9": 50,
        "scene10": 50,
        "scene11": 50,
        "scene12": 50,
        "scene13": 50,
        "scene14": 50,
        "scene15": 50,
        "scene16": 50
    }

    train_catalog = {}
    for i in range(1, 17):
        category = "scene{}".format(i)
        train_catalog[category] = []
        for j in range(train_data_num[category]):
            for k in range(26):
                filename = "{}.{:04d}_view{}".format(category, j, k)
                file_path = os.path.join(data_dir, category, "{}.ply".format(filename))
                if not os.path.exists(file_path):
                    raise Exception("file path not exist: {}".format(file_path))
                train_catalog[category].append(filename)

    test_catalog = {}
    for i in range(1, 17):
        category = "scene{}".format(i)
        test_catalog[category] = []
        for j in range(train_data_num[category], train_data_num[category] + test_data_num[category]):
            for k in range(26):
                filename = "{}.{:04d}_view{}".format(category, j, k)
                file_path = os.path.join(data_dir, category, "{}.ply".format(filename))
                if not os.path.exists(file_path):
                    raise Exception("file path not exist: {}".format(filename))
                test_catalog[category].append("{}.{:04d}_view{}".format(category, j, k))

    # train
    dataset_name = "C3d"
    split_data = {dataset_name: train_catalog}
    split_json = json.dumps(split_data, indent=1)
    dataset_path = "..\\configs\\train_C3d"
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    split_path = os.path.join(dataset_path, "train.json")
    if os.path.isfile(split_path):
        os.remove(split_path)
    with open(split_path, 'w', newline='\n') as f:
        f.write(split_json)

    # test
    split_data = {dataset_name: test_catalog}
    split_json = json.dumps(split_data, indent=1)
    dataset_path = "..\\configs\\test_C3d"
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    split_path = os.path.join(dataset_path, "test.json")
    if os.path.isfile(split_path):
        os.remove(split_path)
    with open(split_path, 'w', newline='\n') as f:
        f.write(split_json)
