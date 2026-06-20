import os
from pathlib import Path

# 数据集路径
images_dir = Path(__file__).parent.parent.parent / "zuiyou_labels" / "zuiyou_images"
labels_dir = Path(__file__).parent.parent.parent / "zuiyou_labels" / "zuiyou_labels"

# 获取所有图片文件
image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
print(f"总图片数量: {len(image_files)}")

# 获取所有非-dehaze的标签文件
label_files = [f for f in labels_dir.glob("*.txt") if "_dehaze" not in f.name]
print(f"总标签数量 (不含_dehaze): {len(label_files)}")

# 检查图片和标签的对应关系
image_names = set([f.stem for f in image_files])
label_names = set([f.stem for f in label_files])

# 找到没有标签的图片
missing_labels = image_names - label_names
if missing_labels:
    print(f"\n警告: {len(missing_labels)} 张图片缺少标签文件")
    print(f"示例: {list(missing_labels)[:5]}")

# 找到没有图片的标签
missing_images = label_names - image_names
if missing_images:
    print(f"\n警告: {len(missing_images)} 个标签文件缺少对应的图片")
    print(f"示例: {list(missing_images)[:5]}")

# 找到匹配的文件
matched = image_names & label_names
print(f"\n匹配的图片-标签对: {len(matched)}")

print("\n建议:")
print("1. 如果匹配数量正确，可以开始训练")
print("2. YOLO将自动划分80%训练集和20%验证集")
print("3. 如果需要手动划分，请创建train和val子目录")
