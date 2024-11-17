import argparse
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--val_rate', type=float, default=0.2)
parser.add_argument('--data', type=str, default='data/datasets')
args = parser.parse_args()


def divide_dataset(i_path, val_p):
    print('正在拆分数据集......')
    image_dir = os.path.join(i_path, 'images')
    txt_dir = os.path.join(i_path, 'labels')
    dir_list = list(os.listdir(image_dir))
    file_list = []
    for filename in dir_list:
        file, ext = os.path.splitext(filename)
        txt_path = os.path.join(txt_dir, file + '.txt')
        if not os.path.exists(txt_path):
            print(f'Warning:{filename}的标签文件不存在，已忽略！')
        else:
            file_list.append(filename)
    if val_p > 0:
        train_list = []
        val_list = []
        train_val_list = []
        n = len(file_list)
        random.shuffle(file_list)
        train_num = n - int(n * val_p)
        for count, path in enumerate(file_list):
            if count < train_num:
                tmp_path = os.path.join(image_dir, path)
                train_list.append(tmp_path)
            else:
                tmp_path = os.path.join(image_dir, path)
                val_list.append(tmp_path)
            train_val_list.append(tmp_path)
        train_txt_path = os.path.join(i_path, 'train.txt')
        val_txt_path = os.path.join(i_path, 'val.txt')
        trainval_txt_path = os.path.join(i_path, 'trainval.txt')
        with open(train_txt_path, 'w') as f:
            for path in train_list:
                f.write(path + '\n')
        with open(val_txt_path, 'w') as f:
            for path in val_list:
                f.write(path + '\n')
        with open(trainval_txt_path, 'w') as f:
            for path in train_val_list:
                f.write(path + '\n')
        print('拆分数据集完毕！')



if __name__ == '__main__':
    divide_dataset(args.data, args.val_rate)
