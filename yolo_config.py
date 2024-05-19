import os
import platform
import shutil
import sys
import torch
import argparse

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from ultralytics import YOLO


class my_YOLO:
    def __init__(self, args):
        self.model_path = args.model
        dirStr, ext = os.path.splitext(self.model_path)
        file = dirStr.split("/")[-1]
        self.file = args.file if args.file else file
        self.model = YOLO(self.model_path)
        self.data = args.data
        self.device = [i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else torch.device(
            'cpu')
        # shutil.rmtree(f'runs/{self.file}', ignore_errors=True)
        os.makedirs(f'runs/{self.file}', exist_ok=True)
