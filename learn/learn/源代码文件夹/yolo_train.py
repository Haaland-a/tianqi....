import sys
import os

# 添加修改过的 ultralytics 到 Python 路径（优先使用本地版本）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, '..', 'ultralytics-main'))

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # 使用绝对路径
    model_yaml = os.path.join(_BASE_DIR, '..', 'ultralytics-main', 'ultralytics', 'cfg', 'models', 'v8', 'yolov8-fca.yaml')
    data_yaml = os.path.join(_BASE_DIR, '..', 'train.yaml')  # 您的数据集配置文件
    pre_model = os.path.join(_BASE_DIR, 'yolov8n.pt')  # 使用本地权重文件

    model = YOLO(model_yaml, task='detect').load(pre_model)

    results = model.train(
        data=data_yaml,
        epochs=190,
        imgsz=640,
        batch=8,
        workers=2,
        augment=True,
        cache=False,
        patience=50,
        save_period=10,
        device='gpu',  # 使用CPU训练（没有GPU）
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
