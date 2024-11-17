import os
import shutil
import time

import cv2

# 图片存储地址
datadir = '../data/datasets/images/'
# 存储训练图片名的txt文件地址
trainlistdir = '../data/datasets/trainval.txt'
# coco格式数据集的train2017目录
traindir = '../data/datasets/train2017/'
shutil.rmtree(traindir, ignore_errors=True)


def bar(des, i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r{}进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur), end="")


def my_move(datadir, trainlistdir, traindir):
    os.makedirs(traindir, exist_ok=True)
    # 打开train.txt文件
    fopen = open(trainlistdir, 'r')
    # 读取图片名称
    file_names = fopen.readlines()
    COUNT = 0
    t1 = time.perf_counter()
    for file_name in file_names:
        file_name = file_name.strip('\n')
        file_name = file_name.replace('\\', '/')
        file_name = file_name.split('/')[-1]
        traindata = datadir + file_name
        if not file_name.endswith('.jpg'):
            img = cv2.imread(traindata)
            # 将图像保存为JPG格式
            file_name_without_ext = file_name.split('.')[0]
            jpg_path = traindir + file_name_without_ext + '.jpg'
            cv2.imwrite(jpg_path, img)
        elif file_name.endswith('.JPG'):
            # 将图像保存为JPG格式
            file_name_without_ext = file_name.split('.')[0]
            jpg_path = traindir + file_name_without_ext + '.jpg'
            shutil.copy(traindata, jpg_path)
        else:
            # 把图片移动至traindir路径下
            # 若想复制可将move改为copy
            shutil.copy(traindata, traindir)
        COUNT += 1
        bar('coco数据移动/清洗', COUNT, len(file_names), t1)
    # 同上


my_move(datadir, trainlistdir, traindir)
