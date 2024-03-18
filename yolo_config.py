import os
import platform
import torch
import argparse

train_yaml = 'data/mydata.yaml'
curpath = os.path.dirname(os.path.realpath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
os_name = str(platform.system())

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="models/yolov8n.yaml")
parser.add_argument('--divide_rate', type=str, default="8:2")
parser.add_argument('--out_file', default=None)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--amp', type=bool, default=True)
parser.add_argument('--is_expo', type=bool, default=True)
args = parser.parse_args()

model = args.model
if os_name == 'Windows':
    num_workers = 0
    dirStr, ext = os.path.splitext(model)
else:
    num_workers = 32
    dirStr, ext = os.path.splitext(model)

file = dirStr.split("/")[-1]
if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    device = [i for i in range(gpu_num)]
else:
    device = torch.device('cpu')

Y_cfg = {
    'model': model,
    'xml_path': f'{curpath}/data/xml_data',
    'divide_in': f'{curpath}/data/i_datasets',
    'divide_out': f'{curpath}/data/datasets',
    'train_divide': args.divide_rate,
    'train_data': train_yaml,
    'out_file': args.out_file if args.out_file else file,
    'device': device,
    'workers': num_workers,
    'epochs': args.epochs,
    'imgsz': args.imgsz,
    'batch': args.batch,
    'detect_in': f'{curpath}/data/i_database',
    'detect_out': f'{curpath}/data/database',
    'AMP': args.amp,

}
