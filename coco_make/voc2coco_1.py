import argparse
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
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


class_dict = {}
class_list = []
parser = argparse.ArgumentParser()
parser.add_argument('--ini_path', default='../data/xml_data')
parser.add_argument('--out_path', default='../data/datasets')

args = parser.parse_args()


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


def pascal_voc_to_coco(xml_path, json_path, image_path):
    global o_num
    global class_dict
    global class_list
    image_width, image_height = get_image_size(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    with open(json_path, 'w', encoding='utf-8') as json_file:
        # time.sleep(0.1)
        c_list = []
        for obj in root.findall('object'):
            name = obj.find('name').text.replace(' ', '')
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            # 转换为YOLOv5格式的坐标
            x, y, w, h = convert_coordinates((image_width, image_height), (xmin, ymin, xmax, ymax))
            # 获取类别索引
            c_dict = {'x': x, 'y': y, 'w': w, 'h': h, 'class': name}
            c_list.append(c_dict)
        json.dump(c_list, json_file)


def convert_voc_to_coco(xml_dir, json_dir, image_dir):
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
        xml_filename = f'{os.path.splitext(filename)[0]}.xml'
        xml_path = os.path.join(xml_dir, xml_filename)
        if not os.path.exists(xml_path):
            print(f"Warning : 图片{filename}对应的xml数据不存在")
            continue
        else:
            true_list.append(filename)
    for filename in true_list:
        xml_filename = f'{os.path.splitext(filename)[0]}.xml'
        xml_path = os.path.join(xml_dir, xml_filename)
        json_filename = f'{os.path.splitext(filename)[0]}.json'
        json_path = os.path.join(json_dir, json_filename)

        image_path = os.path.join(image_dir, filename)

        pascal_voc_to_coco(xml_path, json_path, image_path)
        # shutil.copy(image_path, out_img)
        # shutil.copy(txt_path, out_label)
        # print(f"{xml_path}转化为{txt_filename}")

        COUNT += 1
        bar('coco/json标签转换', COUNT, len(true_list), t1)
    print()


def change_label_main():
    ini_path = args.ini_path
    out_path = args.out_path
    xml_dir = f'{ini_path}/Annotations'
    json_dir = f'{out_path}/json_labels'
    image_dir = f'{ini_path}/Images'
    convert_voc_to_coco(xml_dir, json_dir, image_dir)


if __name__ == '__main__':
    change_label_main()
