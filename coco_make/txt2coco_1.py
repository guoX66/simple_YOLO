import argparse
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET

import yaml
from PIL import Image


def bar(des, i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r{}进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur), end="")


def get_image_size(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height


parser = argparse.ArgumentParser()
parser.add_argument('--class_file', default='./class.yaml')
parser.add_argument('--data_path', default='../data/datasets')
args = parser.parse_args()

with open(args.class_file, 'r', encoding='UTF-8') as f:
    class_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
class_dict = {str(i): class_dict[i] for i in class_dict.keys()}


def convert_coordinates(size, box):
    image_width, image_height = size[0], size[1]
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

    # 归一化处理
    x = (x_min + x_max) / 2.0
    x = int(x)

    y = (y_min + y_max) / 2.0
    y = int(y)

    w = (x_max - x_min)
    w = int(w)

    h = (y_max - y_min)
    h = int(h)
    return x, y, w, h


def pascal_txt_to_coco(txt_path, json_path, image_path):
    image_width, image_height = get_image_size(image_path)

    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        txt_comment = txt_file.readlines()

    with open(json_path, 'w', encoding='utf-8') as json_file:
        # time.sleep(0.1)
        c_list = []
        for lines in txt_comment:
            lines_str = lines.strip().split(' ')
            name = class_dict[lines_str[0]]
            x, y, w, h = lines_str[1:]
            x = int(float(x) * image_width)
            y = int(float(y) * image_height)
            w = int(float(w) * image_width)
            h = int(float(h) * image_height)

            # 获取类别索引
            c_dict = {'x': x, 'y': y, 'w': w, 'h': h, 'class': name}
            c_list.append(c_dict)
        json.dump(c_list, json_file)


def convert_txt_to_coco(txt_dir, json_dir, image_dir):
    # out_img = f'{out_path}/images'
    # out_label = f'{out_path}/labels'
    # os.makedirs(out_img)
    # os.makedirs(out_label)
    COUNT = 0  # 统计次数

    os.makedirs(json_dir, exist_ok=True)
    # 遍历目录中的每个.xml文件并进行转换
    t1 = time.perf_counter()
    dir_list = list(os.listdir(image_dir))
    true_list = []
    for filename in dir_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        if not os.path.exists(txt_path):
            print(f"Warning : 图片{filename}对应的txt数据不存在")
            continue
        else:
            true_list.append(filename)
    for filename in true_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        json_filename = f'{os.path.splitext(filename)[0]}.json'
        json_path = os.path.join(json_dir, json_filename)

        image_path = os.path.join(image_dir, filename)

        pascal_txt_to_coco(txt_path, json_path, image_path)
        # shutil.copy(image_path, out_img)
        # shutil.copy(txt_path, out_label)
        # print(f"{xml_path}转化为{txt_filename}")

        COUNT += 1
        bar('txt/json标签转换', COUNT, len(true_list), t1)
    print()


def change_label_main():
    data_path = args.data_path
    txt_dir = f'{data_path}/labels'
    json_dir = f'{data_path}/json_labels'
    image_dir = f'{data_path}/images'
    convert_txt_to_coco(txt_dir, json_dir, image_dir)


if __name__ == '__main__':
    change_label_main()
    labels = list(class_dict.values())
    labels = ['None'] + labels
    detection_threshold = 0.5
    max_boxes = 200
    js = {
        "detection_threshold": detection_threshold,
        "max_boxes": max_boxes,
        "labels": labels,
    }
    with open(f'class.json', "w", encoding="utf-8") as f:
        json.dump(js, f)
