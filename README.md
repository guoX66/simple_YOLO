# Simple use of YOLO

### The project extends YOLOv8 with some modifications, including dataset production, label conversion, module addition, training inference, etc

### YOLOv8 related content refer to its official website：[Home - Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)



# Ⅰ. Environment deployment

### First install python>=3.10.2

### On a device with nvidia services, run the following command line to see the cuda version

```
nvidia-smi
```

### Install using the following command (X of cu11X should be modified according to cuda version)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### On a device without nvidia services, install using the following command

```bash
pip3 install torch torchvision torchaudio
```

### Run the following command to check the version of torch and cuda

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda)"
```

### Install additional dependencies

```bash
pip install -r requirements
```



# II. Dataset Production

### Dataset for making xml (voc) label files:

[EasyData免费在线标注平台_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1A8411v7MW/?spm_id_from=333.337.search-card.all.click&vd_source=6ede58a42dad439f3299faa3c2d63d9e)

### Labelling site:

https://ai.baidu.com/easydata/app/dataset/list

### The voc datasets should be placed in data/xml_data with the following structure

```
--simple_YOLO
    --data
        --xml_data
            --Annotations
                1.xml
                2.xml
                ...
            --Images
                1.jpg
                2.jpg
                ...
```

### 

# **三、  Adding blocks

### Add blocks to my_block.py and modify parse_model function in ultralytics/nn/tasks accordingly

### Resnext50_32x4d, CBAM、EMA have been added to this project

### resnext50_32x4d reference：[guoX66/YoloPlusPlus (github.com)](https://github.com/guoX66/YoloPlusPlus)

### Now that the yaml model structure has been added and set up, we can run the tests:

```bash
python model_test.py --model models/CBAM1_yolov8n.yaml
```



# Ⅳ. Training

### After the dataset made as Step 2 has been saved to data/xml_data, run the following command to perform label conversion:

```bash
python voc2txt.py 
```

### **If you have a prepared dataset, put the unsplit dataset named i_datasets, and the split dataset named datasets in the data folder with the tag semantic pair set in class.yaml

### To split dataset

```bash
python divide.py --divide_rate 8:2
```

### Run training

```bash
python train.py --model models/yolov8n.yaml --epochs 50 --batch 10 --imgsz 640
```

### When running out of memory, retrain with a smaller batch size

### To enable mixed-precision amp training, run the following command (some versions of cuda will prevent the training process from being recorded, so turn off amp)

```bash
python train.py --model models/yolov8n.yaml --epochs 50 --batch 10 --imgsz 640 --amp True
```



# Ⅴ. Inference

### Put the image you want to detect into your inference file and run the following command to detect it

```bash
python inference.py --task detect --model yolov8n --conf 0.3
```

### The training and detection results are stored in the runs folder
