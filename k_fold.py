# https://blog.csdn.net/qq_45062768/article/details/135111115

import argparse
import datetime
from itertools import chain
import os
from pathlib import Path
import shutil
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r'./data')  # 数据集路径
    parser.add_argument('--ksplit', default=5, type=int)  # K-Fold交叉验证拆分数据集
    parser.add_argument('--im_suffixes', default=['jpg', 'png', 'jpeg'], help='images suffix')  # 图片后缀名
    return parser.parse_args()

def run(func, this_iter, desc="Processing"):
    with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='MyThread') as executor:
        results = list(
            tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc)
        )
    return results

def main(opt):
    dataset_path, ksplit, im_suffixes = Path(opt.data), opt.ksplit, opt.im_suffixes

    save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-Valid')
    save_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图像和标签文件的列表
    images = sorted(list(chain(*[(dataset_path / "images").rglob(f'*.{ext}') for ext in im_suffixes])))
    # images = sorted(image_files)
    labels = sorted((dataset_path / "labels").rglob("*.txt"))

    root_directory = Path.cwd()
    print("当前文件运行根目录:", root_directory)
    if len(images) != len(labels):
        print('*' * 20)
        print('当前数据集和标签数量不一致！！！')
        print('*' * 20)

    # 从YAML文件加载类名
    classes_file = sorted(dataset_path.rglob('mydata.yaml'))[0]
    assert classes_file.exists(), "请创建classes.yaml类别文件"
    if classes_file.suffix == ".txt":
        pass
    elif classes_file.suffix == ".yaml":
        with open(classes_file, 'r', encoding="utf8") as f:
            classes = yaml.safe_load(f)['names']
    cls_idx = sorted(classes.keys())

    # 创建DataFrame来存储每张图像的标签计数
    indx = [l.stem for l in labels]  # 使用基本文件名作为ID（无扩展名）
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    # 计算每张图像的标签计数
    for label in labels:
        lbl_counter = Counter()
        with open(label, 'r') as lf:
            lines = lf.readlines()
        for l in lines:
            # YOLO标签使用每行的第一个位置的整数作为类别
            lbl_counter[int(l.split(' ')[0])] += 1
        labels_df.loc[label.stem] = lbl_counter

    # 用0.0替换NaN值
    labels_df = labels_df.fillna(0.0)

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # 设置random_state以获得可重复的结果
    kfolds = list(kf.split(labels_df))
    folds = [f'split_{n}' for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)

    # 为每个折叠分配图像到训练集或验证集
    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

    # 计算每个折叠的标签分布比例
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # 为避免分母为零，向分母添加一个小值（1E-7）
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio

    ds_yamls = []

    for split in folds_df.columns:
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml.as_posix())
        split_dir = (root_directory / split_dir).as_posix()

        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'train': split_dir + '/train/images',
                'val': split_dir + '/val/images',
                'names': classes
            }, ds_y)
    # print(ds_yamls)
    with open(dataset_path / 'yaml_paths.txt', 'w') as f:
        for path in ds_yamls:
            f.write(path + '\n')

    args_list = [(image, save_path, folds_df) for image in images]

    run(split_images_labels, args_list, desc=f"Creating dataset")

def split_images_labels(args):
    image, save_path, folds_df = args
    label = image.parents[1] / 'labels' / f'{image.stem}.txt'
    if label.exists():
        for split, k_split in folds_df.loc[image.stem].items():
            # 目标目录
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    model = YOLO('models/SLA1_yolov8n.yaml')
    # 从文本文件中加载内容并存储到一个列表中
    ds_yamls = []
    with open(Path(opt.data) / 'yaml_paths.txt', 'r') as f:
        for line in f:
            # 去除每行末尾的换行符
            line = line.strip()
            ds_yamls.append(line)

    # 打印加载的文件路径列表
    print(ds_yamls)

    for k in range(opt.ksplit):
        dataset_yaml = ds_yamls[k]
        name = Path(dataset_yaml).stem
        model.train(
            data=dataset_yaml,
            batch=16,
            epochs=100,
            imgsz=640,
            device=0,
            workers=0,
            project="runs/train_SLA1",
            name=name,)

    print("*"*40)
    print("K-Fold Cross Validation Completed.")
    print("*"*40)

