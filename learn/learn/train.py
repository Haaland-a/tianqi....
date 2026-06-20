import sys
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, 'ultralytics-main'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 您的原有代码...

if __name__ == "__main__":
    model_yaml = os.path.join(_BASE_DIR, 'ultralytics-main', 'ultralytics', 'cfg', 'models', 'v8', 'yolov8-fca.yaml')

    data_yaml = os.path.join(_BASE_DIR, 'train.yaml')
    pre_model = os.path.join(_BASE_DIR, '源代码文件夹', 'yolov8n.pt')

    model = YOLO(model_yaml, task='detect').load(pre_model)

    results = model.train(
        data=data_yaml,
        epochs=190,
        imgsz=640,
        batch=8,
        workers=4,
        augment=True,
        cache=False,
        amp=False,
        patience=50,
        save_period=10,
        device='0',
        project='runs/fca_detect',
        name='train_fca_cl',
        exist_ok=False,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
