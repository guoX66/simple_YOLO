import argparse
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
import yaml
from utils import bar, get_image_size

class_dict = {}
class_list = []
parser = argparse.ArgumentParser()
parser.add_argument('--file', default='mytrain')
parser.add_argument('--ini_path', default='data/xml_data')
parser.add_argument('--out_path', default='data/datasets')
parser.add_argument('--model', type=str, default="yolov8n.pt")
args = parser.parse_args()
dirStr, ext = os.path.splitext(args.model)
file = dirStr.split("/")[-1]
file = args.file if args.file else file


def convert_coordinates(size, box):
    image_width, image_height = size[0], size[1]
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

    # 归一化处理
    x = (x_min + x_max) / (2.0 * image_width)
    x = float('%.6g' % x)

    y = (y_min + y_max) / (2.0 * image_height)
    y = float('%.6g' % y)

    w = (x_max - x_min) / image_width
    w = float('%.6g' % w)

    h = (y_max - y_min) / image_height
    h = float('%.6g' % h)
    return x, y, w, h


def pascal_voc_to_yolov5(xml_path, txt_path, image_path):
    global o_num
    global class_dict
    global class_list
    image_width, image_height = get_image_size(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        # time.sleep(0.1)
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
            if name not in class_list:
                class_list.append(name)
                class_index = class_list.index(name)
                class_dict[class_index] = name
            else:
                class_index = class_list.index(name)
            txt_file.write(f"{class_index} {x} {y} {w} {h}\n")


def convert_voc_to_yolov5(xml_dir, txt_dir, image_dir, out_path):
    # out_img = f'{out_path}/images'
    # out_label = f'{out_path}/labels'
    # os.makedirs(out_img)
    # os.makedirs(out_label)
    COUNT = 0  # 统计次数
    # 创建保存txt文件的目录
    os.makedirs(txt_dir, exist_ok=True)
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
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)

        image_path = os.path.join(image_dir, filename)

        pascal_voc_to_yolov5(xml_path, txt_path, image_path)
        # shutil.copy(image_path, out_img)
        # shutil.copy(txt_path, out_label)
        # print(f"{xml_path}转化为{txt_filename}")

        COUNT += 1
        bar('标签转换', COUNT, len(true_list), t1)
    print()


def change_label_main():
    ini_path = args.ini_path
    dataset_path = args.out_path
    xml_dir = f'{ini_path}/Annotations'
    txt_dir = f'{ini_path}/labels'
    image_dir = f'{ini_path}/Images'

    shutil.rmtree(dataset_path, ignore_errors=True)
    shutil.rmtree(txt_dir, ignore_errors=True)
    os.makedirs(dataset_path)
    convert_voc_to_yolov5(xml_dir, txt_dir, image_dir, dataset_path)
    out_img = f'{dataset_path}/images'
    out_label = f'{dataset_path}/labels'

    shutil.copytree(image_dir, out_img)
    shutil.copytree(txt_dir, out_label)


if __name__ == '__main__':
    change_label_main()
    curpath = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(f'runs/{file}', exist_ok=True)
    with open(f'runs/{file}/class.yaml', "w", encoding="utf-8") as f:
        yaml.dump(class_dict, f)
