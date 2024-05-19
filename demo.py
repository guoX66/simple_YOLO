import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        project='yolov8n',
        data='runs/yolov8n/train.yaml',
        epochs=50,
        imgsz=640,
        workers=8,
        batch=8,
        device=[0],
        amp=True
    )
