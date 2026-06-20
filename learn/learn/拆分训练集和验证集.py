import os
import shutil
import random

# 设置路径 - 使用相对路径
dataset_dir = "../../zuiyou_labels"  # 向上退2级，进入 learn (2) 目录，再找 zuiyou_labels
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

# 获取所有图片文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42)
random.shuffle(image_files)

# 按8:2划分
split_idx = int(len(image_files) * 0.8)
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

# 创建目标目录
for split in ['train', 'val']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# 移动训练集
for img in train_images:
    src_img = os.path.join(images_dir, img)
    dst_img = os.path.join(images_dir, 'train', img)
    shutil.move(src_img, dst_img)

    label_file = img.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
    src_label = os.path.join(labels_dir, label_file)
    if os.path.exists(src_label):
        shutil.move(src_label, os.path.join(labels_dir, 'train', label_file))

# 移动验证集
for img in val_images:
    src_img = os.path.join(images_dir, img)
    dst_img = os.path.join(images_dir, 'val', img)
    shutil.move(src_img, dst_img)

    label_file = img.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
    src_label = os.path.join(labels_dir, label_file)
    if os.path.exists(src_label):
        shutil.move(src_label, os.path.join(labels_dir, 'val', label_file))

print(f"训练集：{len(train_images)} 张")
print(f"验证集：{len(val_images)} 张")