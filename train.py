import shutil
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from ultralytics import YOLO
from yolo_config import Y_cfg, train_yaml, curpath,args
from utils import make_yaml

if __name__ == '__main__':
    make_yaml(curpath, train_yaml)
    model = YOLO(Y_cfg['model'])
    # 训练模型
    output_file = Y_cfg['out_file']
    try:
        shutil.rmtree(f'runs/detect/{output_file}')
    except:
        pass
    results = model.train(name=output_file,
                          data=Y_cfg['train_data'],
                          epochs=Y_cfg['epochs'],
                          imgsz=Y_cfg['imgsz'],
                          workers=Y_cfg['workers'],
                          batch=Y_cfg['batch'],
                          device=Y_cfg['device'],
                          amp=Y_cfg['AMP']
                          )
    if args.is_expo:
        model.export(format='openvino')
        model.export(format='onnx')
        model.export(format='engine', half=True)
