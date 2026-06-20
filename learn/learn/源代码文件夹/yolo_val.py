from ultralytics import YOLO

if __name__ == "__main__":
    # 加载自定义训练好的模型
    pth_path = "yolov8n.pt"
    model = YOLO(pth_path)

    # 对模型进行验证
    metrics = model.val()

    # 输出各种性能指标
    print("mAP@50-95:", metrics.box.map)
    print("mAP@50:", metrics.box.map50)
    print("mAP@75:", metrics.box.map75)
    print("mAP of each category:", metrics.box.maps)



