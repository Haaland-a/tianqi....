from ultralytics import YOLO
import os
from PIL import Image

# 加载预训练的YOLOv8模型
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(_BASE_DIR, '源代码文件夹', 'yolov8n.pt'))

# 无标签图片所在的文件夹路径
image_folder = r'D:\数据\H\train\train\hazy_images'

# 保存伪标签的文件夹路径
label_folder = r'D:\数据\H\train\train\lables'

# 创建保存伪标签的文件夹
if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# 遍历图片文件夹中的所有图片
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(image_folder, filename)
        # 打开图片
        image = Image.open(image_path)
        # 使用模型进行预测
        results = model.predict(image)
        # 获取预测结果
        for result in results:
            boxes = result.boxes.cpu().numpy()
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_folder, label_filename)
            with open(label_path, 'w') as f:
                for box in boxes:
                    cls = int(box.cls[0])
                    x_center = box.xywhn[0][0]
                    y_center = box.xywhn[0][1]
                    width = box.xywhn[0][2]
                    height = box.xywhn[0][3]
                    confidence = box.conf[0]
                    # 这里可以根据置信度进行筛选
                    if confidence > 0.5:
                        line = f"{cls} {x_center} {y_center} {width} {height}\n"
                        f.write(line)
