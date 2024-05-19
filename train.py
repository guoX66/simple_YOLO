import argparse
import shutil
import os
import sys
import yaml

from yolo_config import my_YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="yolov8n.pt")
parser.add_argument('--data', type=str, default='data/datasets')
parser.add_argument('--file', default='mytrain')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--amp', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()


class my_Train(my_YOLO):
    def __init__(self, args):
        super().__init__(args)
        self.epochs = args.epochs
        self.batch = args.batch
        self.imgsz = args.imgsz
        self.amp = args.amp
        self.cur_path = os.path.dirname(os.path.abspath(__file__))

    def make_yaml(self):
        train_yaml = f'{self.cur_path}/runs/{self.file}/train.yaml'
        class_yaml = f'{self.cur_path}/runs/{self.file}/class.yaml'
        with open(class_yaml, 'r', encoding='UTF-8') as f:
            class_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        class_dict = {int(i): class_dict[i] for i in class_dict.keys()}

        desired_caps = {
            'path': f'{self.cur_path}/{self.data}',  # dataset root dir
            'train': 'train.txt',
            'val': 'val.txt',
            # Classes
            'names': class_dict
        }

        with open(train_yaml, "w", encoding="utf-8") as f:
            yaml.dump(desired_caps, f)

    def start_train(self):
        self.model.train(
            project=f'runs/{self.file}',
            data=f'{self.cur_path}/runs/{self.file}/train.yaml',
            epochs=self.epochs,
            imgsz=self.imgsz,
            workers=args.num_workers,
            batch=self.batch,
            device=self.device,
            amp=self.amp
        )

    def export(self):
        self.model.export(format='openvino')
        self.model.export(format='onnx')
        self.model.export(format='engine', half=True)


if __name__ == '__main__':
    model = my_Train(args)
    model.make_yaml()
    model.start_train()
