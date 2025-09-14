# EYE-yolov10s
# Project Introduction
Research on the Application of the Improved YOLOv10s Algorithm in Eye Disease Detection

# Installation
```
conda create -n yolo python=3.9  
conda activate yolo  
pip install -r requirements.txt  
pip install -e .
```
# Demo
Run the following command to start the project:  
```
python app.py
```
# Catalog Structure Description
```
EYE-yolov10s/  
├── ultralytics/  
│   ├── assets/        # Resource files and model assets  
│   ├── cfg/           # Configuration files  
│   ├── data/          # Dataset and dataloader definitions  
│   ├── engine/        # Training, validation, and prediction pipelines  
│   ├── hub/           # Model downloading and registry logic  
│   ├── models/        # Model architecture definitions 
│   ├── nn/            # Custom layers and loss functions  
│   ├── solutions/     # End-to-end solution pipelines  
│   ├── trackers/      # Tracking-related modules  
│   ├── utils/         # Utility functions and evaluation metrics  
│   └── __init__.py    # Package initializer  
│  
├── datasets/ # Data preparation and preprocessing scripts  
│  
├── runs/ # Training logs and saved model checkpoints  
│  
├── app.py # Entry point for training  
├── requirements.txt # Python dependencies  
└── README.md # Project introduction and usage  
```
# Data Preparation
```
datasets/  
└── dataset_eye/  
    ├── images/  
    │   ├── train/  
    │   ├── val/  
    │   └── test/  
    ├── labels/  
    │   ├── train/  
    │   ├── val/  
    │   └── test/  
    └── data.yaml   # Dataset configuration file for DOTA (class names, paths, etc.)
```
# Validation
```
yolo val model=path/weight.pth data=path/data.yaml batch=1
```
# Or
```
from ultralytics import YOLOv10

model = YOLOv10('your-model.pt')

model.val(data='coco.yaml', batch=256)
```
# Training
```
yolo detect train data=path/data.yaml model=path/eye-yolov10s.yaml epochs=500 batch=8 imgsz=640

```
# Or
```
from ultralytics import YOLOv10

model = YOLOv10()


model.train(data='path/data.yaml', epochs=500, batch=8, imgsz=640)
```
# Prediction
```
yolo predict model=path/your-model.pth source=path/data.yaml
```
# Or
```
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('path/your-model.pth')

model.predict(source='path/data.yaml',)
```
