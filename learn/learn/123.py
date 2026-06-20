import os
import shutil


def add_prefix_to_labels(label_dir, output_dir, prefix="foggy_"):
    """
    为YOLO格式的txt标注文件添加前缀，使其与图像文件名保持一致

    参数:
    label_dir: 原始标注文件(.txt)所在目录
    output_dir: 转换后标注文件的输出目录
    prefix: 需要添加的前缀，如"foggy_"
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有txt文件
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt") and filename != "classes.txt":  # 排除类别文件
            input_path = os.path.join(label_dir, filename)
            # 构建新文件名，添加前缀
            new_filename = f"{prefix}{filename}"
            output_path = os.path.join(output_dir, new_filename)

            # 复制文件内容到新文件
            with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
                f_out.writelines(f_in.readlines())

            print(f"已重命名: {filename} -> {new_filename}")

    # 复制类别文件（如果存在）
    classes_path = os.path.join(label_dir, "classes.txt")
    if os.path.exists(classes_path):
        shutil.copy(classes_path, os.path.join(output_dir, "classes.txt"))
        print("已复制类别文件")


# 使用示例
add_prefix_to_labels(
    label_dir= r"D:\tomcatz\labelsyolov\labelsyolov\val",  # 原始标签文件夹
    output_dir= r"D:\tomcatz\label2",  # 输出文件夹
    prefix="foggy_"  # 前缀，需与图像文件名前缀一致
)