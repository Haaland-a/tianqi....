import os
import shutil
from PIL import Image
import random
import numpy as np


# 统一图片尺寸和格式
def resize_images(input_folder, output_folder, size=(224, 224), target_format='.jpg'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path) and (filename.endswith('.jpg') or filename.endswith('.png')):
            img = Image.open(img_path)
            img = img.resize(size)
            new_filename = os.path.splitext(filename)[0] + target_format
            img.save(os.path.join(output_folder, new_filename))


# 合并标签文件
def merge_annotations(your_annotation_folder, others_annotation_folder, output_annotation_folder):
    if not os.path.exists(output_annotation_folder):
        os.makedirs(output_annotation_folder)

    all_annotation_files = []
    for folder in [your_annotation_folder, others_annotation_folder]:
        for filename in os.listdir(folder):
            annotation_path = os.path.join(folder, filename)
            if os.path.isfile(annotation_path) and (filename.endswith('.txt') or filename.endswith('.xml')):
                all_annotation_files.append(annotation_path)

    for annotation_file in all_annotation_files:
        try:
            shutil.copy(annotation_file, output_annotation_folder)
            print(f"成功复制文件: {annotation_file}")
        except Exception as e:
            print(f"复制文件 {annotation_file} 时出错: {e}")


# 整理图片和对应标签关系
def get_image_label_pairs(image_folder, annotation_folder):
    image_label_pairs = []
    for img_filename in os.listdir(image_folder):
        img_base_name = os.path.splitext(img_filename)[0]
        # 尝试查找对应的txt和xml标签文件
        for ext in ['.txt', '.xml']:
            annotation_filename = img_base_name + ext
            annotation_path = os.path.join(annotation_folder, annotation_filename)
            img_path = os.path.join(image_folder, img_filename)
            if os.path.isfile(img_path):
                print(f"图片文件存在: {img_path}")
            else:
                print(f"图片文件不存在: {img_path}")
            if os.path.isfile(annotation_path):
                img_path = os.path.join(image_folder, img_filename)
                image_label_pairs.append((img_path, annotation_path))
                print(f"找到匹配的图片和标签: {img_path} - {annotation_path}")
                break
        else:
            print(f"未找到匹配的标签文件: {img_filename}")
    return image_label_pairs


# 划分数据集
def split_datasets(image_label_pairs, test_size=0.2, val_size=0.1, random_state=42):
    np.random.seed(random_state)
    random.shuffle(image_label_pairs)
    total_size = len(image_label_pairs)
    test_size = int(total_size * test_size)
    val_size = int(total_size * val_size)
    train_size = total_size - test_size - val_size

    train_data = image_label_pairs[:train_size]
    val_data = image_label_pairs[train_size:train_size + val_size]
    test_data = image_label_pairs[train_size + val_size:]

    return train_data, val_data, test_data


# 示例使用
your_image_folder = r'D:\数据\ss\images'
others_image_folder = r'D:\tomcatz\image2'
your_annotation_folder = r'D:\数据\ss\labels'
others_annotation_folder = r'D:\tomcatz\label2'

# 融合后的主文件夹
merged_dataset_folder = r'D:\数据\dataset'
# 融合后图片和标签的子文件夹
merged_images_folder = os.path.join(merged_dataset_folder, 'images')
merged_labels_folder = os.path.join(merged_dataset_folder, 'labels')

# 统一图片尺寸和格式并复制到融合后的图片文件夹
resize_images(your_image_folder, merged_images_folder)
resize_images(others_image_folder, merged_images_folder)

# 合并标签文件到融合后的标签文件夹
merge_annotations(your_annotation_folder, others_annotation_folder, merged_labels_folder)

# 整理图片和对应标签关系
all_image_label_pairs = get_image_label_pairs(merged_images_folder, merged_labels_folder)

# 划分数据集
train_data, val_data, test_data = split_datasets(all_image_label_pairs)

print("训练集数据数量:", len(train_data))
print("验证集数据数量:", len(val_data))
print("测试集数据数量:", len(test_data))
