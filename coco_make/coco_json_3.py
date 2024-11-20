import argparse
import json
import glob
import time

import cv2 as cv
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='mytrain')
args = parser.parse_args()
file = args.file

# 现有的标注文件地址
label_path = '../data/datasets/json_labels/'
# 保存地址
save_path = '../data/datasets/annotations'
os.makedirs(save_path, exist_ok=True)

class_yaml = f'../runs/{file}/class.yaml'
with open(class_yaml, 'r', encoding='UTF-8') as f:
    class_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
class_dict = {class_dict[i]: i + 1 for i in class_dict.keys()}
img_id = 0


def bar(des, i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r{}进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur), end="")


class tococo(object):
    def __init__(self, jpg_paths, label_path, save_path, root_path):
        self.images = []
        self.categories = []
        self.annotations = []
        # 返回每张图片的地址
        self.jpgpaths = jpg_paths
        self.save_path = save_path
        self.label_path = label_path
        # 可根据情况设置类别，这里只设置了一类
        self.class_ids = class_dict
        self.coco = {}
        self.root_path = root_path

    def npz_to_coco(self):
        global img_id
        annid = 0
        COUNT = 0
        t1 = time.perf_counter()
        for old_img_path in self.jpgpaths:
            old_img_path = old_img_path.replace('\\', '/')
            old_imgname = old_img_path.split('/')[-1].split('.')[0]
            imgname = str(img_id).zfill(12) + ".jpg"
            img_path = os.path.join(f'../data/datasets/{self.root_path}', imgname)
            os.rename(old_img_path, img_path)

            img = cv.imread(img_path)
            jsonf = open(self.label_path + old_imgname + '.json').read()  # 读取json
            labels = json.loads(jsonf)
            h, w = img.shape[:-1]
            self.images.append(self.get_images(imgname, h, w, img_id))
            for label in labels:
                # self.categories.append(self.get_categories(label['class'], self.class_id))
                px, py, pw, ph = label['x'], label['y'], label['w'], label['h']
                box = [px, py, pw, ph]
                # print(box)
                self.annotations.append(self.get_annotations(box, img_id, annid, label['class']))
                annid = annid + 1
            img_id = img_id + 1
            COUNT += 1
            bar('coco数据整合', COUNT, len(self.jpgpaths), t1)
        print()
        self.coco["images"] = self.images
        self.categories.append(self.get_categories(label['class'], self.class_ids[label['class']]))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations
        # print(self.coco)

    def get_images(self, filename, height, width, image_id):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        # 文件名加后缀
        image["file_name"] = filename
        # print(image)
        return image

    def get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "Positive Cell"
        # id=0
        category['id'] = class_id
        # name=1
        category['name'] = name
        # print(category)
        return category

    def get_annotations(self, box, image_id, ann_id, class_name):
        annotation = {}
        w, h = box[2], box[3]
        area = w * h
        annotation['segmentation'] = [[]]
        annotation['iscrowd'] = 0
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = float(area)
        # category_id=0
        annotation['category_id'] = self.class_ids[class_name]
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        return annotation

    def save_json(self):
        self.npz_to_coco()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        f = open(os.path.join(save_path + f'/instances_{self.root_path}.json'), 'w')
        f.write(instances_train2017)
        f.close()


# 可改为train2017，要对应上面的
root_path = 'train2017'
jpg_paths = glob.glob(f'../data/datasets/{root_path}/*.jpg')
c = tococo(jpg_paths, label_path, save_path, root_path)
c.save_json()

# root_path = 'val2017'
# jpg_paths = glob.glob(f'../data/datasets/{root_path}/*.jpg')
#
# c = tococo(jpg_paths, label_path, save_path, root_path)
# c.save_json()
