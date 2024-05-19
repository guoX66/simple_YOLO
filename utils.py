import os
import time
from PIL import Image
import yaml


def is_file_empty(filename):
    # 获取文件的完整路径
    file_path = os.path.abspath(filename)

    # 获取文件大小
    file_size = os.path.getsize(file_path)

    return file_size == 0


def get_image_size(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height


def bar(des, i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r{}进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur), end="")


def read_cfg(base_dir):
    path = os.path.join(base_dir, 'Cfg.yaml')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            Cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        tr_Cfg = Cfg['train']
        Ir_Cfg = Cfg['inference']
        ba_Cfg = Cfg['base']
        return tr_Cfg, Ir_Cfg, ba_Cfg, Cfg
    else:
        raise FileNotFoundError('Cfg.yaml not found')


def make_yaml(curpath, train_yaml, class_yaml='class.yaml'):
    with open(class_yaml, 'r', encoding='UTF-8') as f:
        class_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    class_dict = {int(i): class_dict[i] for i in class_dict.keys()}
    desired_caps = {
        'path': f'{curpath}/data/i_datasets',  # dataset root dir
        'train': 'train.txt',
        'val': 'val.txt',
        # Classes
        'names': class_dict
    }
    with open(train_yaml, "w", encoding="utf-8") as f:
        yaml.dump(desired_caps, f)
