import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose model path')
    parser.add_argument('--model', type=str, default=f'yolov8n',
                        help='trained model path')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--task', type=str, default='detect', choices=('detect', 'segment'), help='task')
    parser.add_argument('--source', type=str, default='inference')
    args = parser.parse_args()
    model = YOLO(os.path.join(current_dir, f'runs/{args.task}/{args.model}/weights/best.pt'))
    result = model(args.source, save=True, conf=args.conf)
