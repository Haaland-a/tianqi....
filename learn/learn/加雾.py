import cv2
import numpy as np


def add_fog_effect(image_path, output_path, fog_intensity=0.5):
    # 读取图像
    img = cv2.imread(image_path)
    h, w, c = img.shape

    # 生成随机透射率（模拟雾浓度）
    mask = np.random.uniform(low=0.3, high=1.0 - fog_intensity, size=(h, w))
    # 大气光值（模拟雾的颜色，可调整为灰色/白色）
    A = np.array([200, 200, 200], dtype=np.uint8)

    # 雾效公式：I = J*t + A*(1-t)
    foggy_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(3):
        foggy_img[:, :, i] = img[:, :, i] * mask + A[i] * (1 - mask)

    # 保存结果
    cv2.imwrite(output_path, foggy_img)
    print(f"雾效图像已保存至：{output_path}")


# 批量处理示例（遍历文件夹内所有图像）
import os

input_dir = r"D:\tomcatz\imagesyolov\imagesyolov\train" # 清晰图像文件夹
output_dir = r"D:\tomcatz\image1"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"foggy_{filename}")
        add_fog_effect(input_path, output_path, fog_intensity=np.random.uniform(0.2, 0.8))