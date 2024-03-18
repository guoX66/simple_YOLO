import argparse

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ultralytics import YOLO

parser = argparse.ArgumentParser(description='choose model path')
parser.add_argument('--model', type=str, default=f'models/CBAM1_yolov8n.yaml',
                    help='trained model path')
parser.add_argument('--source', type=str, default='assets/bus.jpg')
args = parser.parse_args()

model = YOLO(args.model, verbose=True)
result = model(args.source)
