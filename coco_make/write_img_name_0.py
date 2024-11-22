import argparse
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/datasets')
args = parser.parse_args()


def divide_dataset(i_path):
    print('正在写入图片路径......')
    image_dir = i_path + '/images'
    txt_dir = i_path + '/labels'
    dir_list = list(os.listdir(image_dir))
    file_list = []
    for filename in dir_list:
        file, ext = os.path.splitext(filename)
        txt_path = txt_dir + '/' + file + '.txt'
        if not os.path.exists(txt_path):
            print(f'Warning:{filename}的标签文件不存在，已忽略！')
        else:
            file_list.append(filename)


    train_val_list = []

    for count, path in enumerate(file_list):
        tmp_path = image_dir + '/' + path
        train_val_list.append(tmp_path)

    trainval_txt_path = os.path.join(i_path, 'trainval.txt')
    with open(trainval_txt_path, 'w') as f:
        for path in train_val_list:
            f.write(path + '\n')
    print('写入图片路径完毕！')


if __name__ == '__main__':
    divide_dataset(args.data,)
