from ultralytics import YOLO

import os

if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(_BASE_DIR, '..', 'runs', 'detect', 'train10', 'weights', 'best.pt')

    test_path = os.path.join(_BASE_DIR, '..', '..', '..', 'multi_weather_dataset', 'fog', 'images')
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=0.5)  # predict on an image

