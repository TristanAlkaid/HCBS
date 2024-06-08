import os
import pickle
import re

def get_file_list(folder_path="data_1/content_sentence"):
    """列出文件夹下的所有文件和子文件夹"""
    # 使用os.listdir()列出文件夹下的所有文件和子文件夹
    contents = os.listdir(folder_path)

    # 仅保留文件，排除子文件夹
    files = [f for f in contents]

    for i in range(len(files)):
        files[i] = "football/" + files[i]

    # 现在，'files'列表包含了文件夹下的所有文件名字
    return files


def clean_train_videos(data):
    files = get_file_list()
    # 使用列表推导式过滤列表B中不含列表A元素字符的元素
    flag = [item for item in data["train_videos"][0] if item in files]
    data["train_videos"][0] = flag
    return data


def clean_test_videos(data):
    files = get_file_list()
    # 使用列表推导式过滤列表B中不含列表A元素字符的元素
    flag = [item for item in data["test_videos"][0] if item in files]
    data["test_videos"][0] = flag
    return data


def clean_nframes(data):
    files = get_file_list()
    flag = {key: value for key, value in data["nframes"].items() if key in files}
    data["nframes"] = flag
    return data


def clean_resolution(data):
    files = get_file_list()
    flag = {key: value for key, value in data["resolution"].items() if key in files}
    data["resolution"] = flag
    return data


def clean_gttubes(data):
    files = get_file_list()
    flag = {key: value for key, value in data["gttubes"].items() if key in files}
    data["gttubes"] = flag
    return data


def write_pkl_to_txt(data, path="data/multisports/multisports_GT.txt"):
    with open(path, "w") as f:
        f.write(str(data))


def save_football_labels(data):
    flag = [item for item in data["labels"] if 'football' in item]
    data["labels"] = flag
    return data


def save_football_train_videos(data):
    flag = [item for item in data["train_videos"][0] if 'football' in item]
    data["train_videos"][0] = flag
    return data


def save_football_test_videos(data):
    flag = [item for item in data["test_videos"][0] if 'football' in item]
    data["test_videos"][0] = flag
    return data


def save_football_nframes(data):
    flag = {key: value for key, value in data["nframes"].items() if 'football' in key}
    data["nframes"] = flag
    return data


def save_football_resolution(data):
    flag = {key: value for key, value in data["resolution"].items() if 'football' in key}
    data["resolution"] = flag
    return data


def save_football_gttubes(data):
    flag = {key: value for key, value in data["gttubes"].items() if 'football' in key}
    data["gttubes"] = flag

    for vid, labels in data["gttubes"].items():
        dict = {}
        for key, labels in labels.items():
            key = key - 33
            if key < 0 or key > 15:
                print("error")
            dict.update({key: labels})
        data["gttubes"][vid] = dict

    return data


if __name__ == "__main__":
    # 指定.pkl文件的路径
    file_path = 'data/multisports/multisports_GT.pkl'

    # 使用pickle.load()来加载.pkl文件
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    print(len(data["test_videos"][0]))
    print(len(data["train_videos"][0]))

    # data = save_football_labels(data)
    # data = save_football_train_videos(data)
    # data = save_football_test_videos(data)
    # data = save_football_nframes(data)
    # data = save_football_resolution(data)
    # data = save_football_gttubes(data)
    #
    # data = clean_train_videos(data)
    # data = clean_test_videos(data)
    # data = clean_nframes(data)
    # data = clean_resolution(data)
    # data = clean_gttubes(data)
    # # 保存修改后的数据
    # with open('data/cleaned/multisports_GT.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    #     print("保存成功")
    #
    write_pkl_to_txt(data)
