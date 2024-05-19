import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='choose model path')
parser.add_argument('--file', default='mytrain')
parser.add_argument('--train_file', default='train')
parser.add_argument('--conf', type=float, default=0.3)
parser.add_argument('--source', type=str, default='inference')
args = parser.parse_args()

if __name__ == '__main__':
    model = YOLO(os.path.join(current_dir, f'runs/{args.file}/{args.train_file}/weights/best.pt'))
    result = model(args.source, save=True, conf=args.conf)
