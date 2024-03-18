import os
import random
import shutil
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from yolo_config import Y_cfg


def divide_dataset(path, o_path, train_p, val_p):
    print('正在拆分数据集......')
    shutil.rmtree(o_path, ignore_errors=True)
    train_path_i = os.path.join(o_path, 'train', 'images')
    train_path_l = os.path.join(o_path, 'train', 'labels')
    val_path_i = os.path.join(o_path, 'val', 'images')
    val_path_l = os.path.join(o_path, 'val', 'labels')


    os.makedirs(train_path_i)
    os.makedirs(train_path_l)
    os.makedirs(val_path_i)
    os.makedirs(val_path_l)

    image_dir = os.path.join(path, 'images')
    txt_dir = os.path.join(path, 'labels')
    dir_list = list(os.listdir(image_dir))
    true_list = []
    background_list = []
    for filename in dir_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        img_path = os.path.join(image_dir, filename)
        txt_path = os.path.join(txt_dir, txt_filename)
        if not os.path.exists(txt_path):
            background_list.append(filename)
            shutil.copy(img_path, train_path_i)
        else:
            true_list.append(filename)

    n = len(true_list)
    random.shuffle(true_list)
    train_list = true_list[:int(n * train_p / 10)]
    val_list = true_list[int(n * train_p / 10):]

    for filename in train_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        img_path = os.path.join(image_dir, filename)
        shutil.copy(img_path, train_path_i)
        shutil.copy(txt_path, train_path_l)

    for filename in val_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        img_path = os.path.join(image_dir, filename)
        shutil.copy(img_path, val_path_i)
        shutil.copy(txt_path, val_path_l)
    print('拆分完毕！')


if __name__ == '__main__':
    t, v = int(Y_cfg['train_divide'][0]), int(Y_cfg['train_divide'][2])
    divide_dataset(Y_cfg['divide_in'], Y_cfg['divide_out'], t, v)
