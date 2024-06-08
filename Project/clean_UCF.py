import pickle

def write_pkl_to_txt(data, path="data/cleaned/multisports_GT.txt"):
    with open(path, "w") as f:
        f.write(str(data))

def save_football_labels(data):
    flag = [item for item in data["labels"] if 'SoccerJuggling' in item]
    data["labels"] = flag
    return data


def save_football_gttubes(data):
    flag = {key: value for key, value in data["gttubes"].items() if 'SoccerJuggling' in key}
    data["gttubes"] = flag

    for vid, labels in data["gttubes"].items():
        dict = {}
        for key, labels in labels.items():
            key = 0
            if not key == 0:
                print("error")
            dict.update({key: labels})
        data["gttubes"][vid] = dict

    return data


def save_football_nframes(data):
    flag = {key: value for key, value in data["nframes"].items() if 'SoccerJuggling' in key}
    data["nframes"] = flag
    return data


def save_football_train_videos(data):
    flag0 = [item for item in data["train_videos"][0] if 'SoccerJuggling' in item]
    data["train_videos"][0] = flag0

    return data


def save_football_test_videos(data):
    flag0 = [item for item in data["test_videos"][0] if 'SoccerJuggling' in item]
    data["test_videos"][0] = flag0

    return data

def save_football_resolution(data):
    flag = {key: value for key, value in data["resolution"].items() if 'SoccerJuggling' in key}
    data["resolution"] = flag
    return data


if __name__ == "__main__":
    # 指定.pkl文件的路径
    file_path = 'data/UCF101_v2/UCF101v2-GT.pkl'

    # 使用pickle.load()来加载.pkl文件
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')


    data = save_football_labels(data)
    data = save_football_train_videos(data)
    data = save_football_test_videos(data)
    data = save_football_nframes(data)
    data = save_football_resolution(data)
    data = save_football_gttubes(data)

    # 保存修改后的数据
    with open('data/UCF101_v2/UCF101v2-GT.pkl', 'wb') as file:
        pickle.dump(data, file)
        print("保存成功")

    write_pkl_to_txt(data, path="data/UCF101_v2/UCF101v2-GT.txt")
