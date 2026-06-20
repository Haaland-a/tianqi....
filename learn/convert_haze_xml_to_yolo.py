"""
雾霾数据集 XML 转 YOLO 格式脚本
功能：
1. 解析 XML 标签
2. 转换为 YOLO TXT 格式
3. 根据图片数量重新编号并匹配
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil


def class_name_to_id(class_name):
    """将类别名称转换为ID（与你的项目保持一致）"""
    class_mapping = {
        'bicycle': 0,
        'bike': 0,
        'motorcycle': 1,
        'motorbike': 1,
        'car': 2,
        'person': 3,
        'pedestrian': 3
    }
    return class_mapping.get(class_name.lower(), None)


def convert_xml_to_yolo(xml_file, img_width, img_height):
    """
    解析XML文件并转换为YOLO格式
    
    Returns:
        list: YOLO格式的标注列表 [(class_id, x_center, y_center, w, h), ...]
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片尺寸（优先使用XML中的，如果有的话）
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            width = img_width
            height = img_height
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            
            class_name = name_elem.text
            class_id = class_name_to_id(class_name)
            
            if class_id is None:
                print(f"  ⚠️  未知类别: {class_name}")
                continue
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 转换为YOLO格式
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            yolo_annotations.append((class_id, x_center, y_center, box_width, box_height))
        
        return yolo_annotations
    
    except Exception as e:
        print(f"  ❌ 解析XML失败: {e}")
        return []


def write_yolo_label(annotations, output_path):
    """写入YOLO格式的标签文件"""
    with open(output_path, 'w') as f:
        for class_id, x_center, y_center, w, h in annotations:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def get_image_dimensions(img_path):
    """
    获取图片尺寸
    使用PIL处理中文路径问题
    """
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except ImportError:
        print("❌ 请安装 Pillow: pip install Pillow")
        return None, None


def main():
    """主函数"""
    print("=" * 70)
    print("雾霾数据集 XML 转 YOLO 格式工具")
    print("=" * 70)
    
    # 配置路径
    base_dir = Path(__file__).parent.parent / "multi_weather_dataset" / "haze"
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # 获取所有图片文件
    image_files = sorted([
        f for f in images_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    # 获取所有XML文件
    xml_files = sorted(labels_dir.glob("*.xml"))
    
    print(f"\n📊 数据统计:")
    print(f"  图片数量: {len(image_files)}")
    print(f"  XML标签数量: {len(xml_files)}")
    
    if len(image_files) != len(xml_files):
        print(f"\n⚠️  警告: 图片和标签数量不匹配！")
        print(f"  将使用前 {min(len(image_files), len(xml_files))} 个进行配对")
    
    # 确定处理数量（只处理有对应XML的图片）
    process_count = min(len(image_files), len(xml_files))
    
    if process_count == 0:
        print("\n❌ 没有可处理的文件！")
        return
    
    print(f"\n✅ 将处理前 {process_count} 个样本")
    
    # 创建备份目录
    backup_labels_dir = base_dir / "labels_xml_backup"
    backup_labels_dir.mkdir(exist_ok=True)
    
    # 移动XML文件到备份目录
    print(f"\n【步骤1】备份XML文件...")
    for xml_file in xml_files:
        shutil.move(str(xml_file), str(backup_labels_dir / xml_file.name))
    print(f"  ✓ XML文件已备份到: {backup_labels_dir}")
    
    # 转换并生成YOLO标签
    print(f"\n【步骤2】转换标签并重新编号...")
    success_count = 0
    
    for i in range(process_count):
        img_file = image_files[i]
        xml_file = backup_labels_dir / f"{i+1}.xml"
        
        # 新的文件名（统一编号）
        new_name = f"haze_{i+1:04d}"
        new_img_name = f"{new_name}{img_file.suffix}"
        new_lbl_name = f"{new_name}.txt"
        
        try:
            # 重命名图片
            new_img_path = images_dir / new_img_name
            shutil.move(str(img_file), str(new_img_path))
            
            # 检查对应的XML是否存在
            if xml_file.exists():
                # 获取图片尺寸
                img_width, img_height = get_image_dimensions(new_img_path)
                
                if img_width and img_height:
                    # 转换XML为YOLO格式
                    annotations = convert_xml_to_yolo(xml_file, img_width, img_height)
                    
                    if annotations:
                        # 写入YOLO标签
                        lbl_path = labels_dir / new_lbl_name
                        write_yolo_label(annotations, lbl_path)
                        success_count += 1
                        
                        if (i + 1) % 50 == 0:
                            print(f"  已处理: {i+1}/{process_count} (当前: {new_name})")
                    else:
                        print(f"  ⚠️  {new_name}: 无有效标注")
                else:
                    print(f"  ❌ {new_name}: 无法获取图片尺寸")
            else:
                print(f"  ⚠️  {new_name}: 缺少XML标签文件")
        
        except Exception as e:
            print(f"  ❌ 处理 {img_file.name} 时出错: {e}")
    
    # 最终统计
    print("\n" + "=" * 70)
    print(f"✅ 转换完成！")
    print(f"\n📊 最终统计:")
    final_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    final_labels = len(list(labels_dir.glob("*.txt")))
    print(f"  图片: {final_images} 张")
    print(f"  标签: {final_labels} 个")
    print(f"  成功配对: {success_count} 个")
    
    # 检查是否有未处理的图片
    unprocessed_images = [f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not f.name.startswith('haze_')]
    if unprocessed_images:
        print(f"\n⚠️  发现 {len(unprocessed_images)} 个未处理的图片（缺少对应XML标签）")
        print(f"  建议删除这些图片或手动添加标签")
    
    if final_images == final_labels and final_images > 0:
        print("\n✅ 数据集格式正确，可以开始训练！")
    else:
        print(f"\n⚠️  警告: 图片和标签数量不匹配")
        print(f"  提示: 运行以下命令删除多余图片:")
        print(f"  python remove_unmatched_images.py")
    
    print("=" * 70)
    
    # 显示示例
    print(f"\n📝 示例文件:")
    sample_images = sorted(list(images_dir.glob("*.jpg")))[:3]
    for img in sample_images:
        print(f"  图片: {img.name}")
        lbl = labels_dir / img.with_suffix('.txt').name
        if lbl.exists():
            print(f"  标签: {lbl.name}")
            with open(lbl, 'r') as f:
                lines = f.readlines()
                print(f"    标注数: {len(lines)}")
                if lines:
                    print(f"    示例: {lines[0].strip()}")
        print()


if __name__ == "__main__":
    main()
