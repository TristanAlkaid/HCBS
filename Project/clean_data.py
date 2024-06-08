import os


def get_file_list(folder_path):
    """列出文件夹下的所有文件和子文件夹"""
    # 使用os.listdir()列出文件夹下的所有文件和子文件夹
    contents = os.listdir(folder_path)

    # 仅保留文件，排除子文件夹
    files = [f for f in contents]

    for i in range(len(files)):
        files[i] = files[i] + ".mp4"

    # 现在，'files'列表包含了文件夹下的所有文件名字
    return files

def remove_file(folder_path, files_to_keep):
    # 使用os.listdir()列出文件夹下的所有文件
    all_files = os.listdir(folder_path)

    # 要删除的文件
    files_to_delete = [f for f in all_files if f not in files_to_keep]

    # 删除不需要的文件
    for file_to_delete in files_to_delete:
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)
        print(f"Deleted: {file_to_delete}")


path = "/home/zxy/code/MOC-Detector/data/content_sentence"
files = get_file_list(path)

target_path = "/home/zxy/code/MOC-Detector/data/football"
remove_file(target_path, files)