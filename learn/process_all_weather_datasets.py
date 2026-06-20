"""
多天气数据集统一处理脚本
处理雨天和雪天数据集，转换为YOLO格式并整合到multi_weather_dataset
"""
import os
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import shutil


def class_name_to_id(class_name):
    """将类别名称转换为ID（与项目保持一致）"""
    class_mapping = {
        'bicycle': 0,
        'bike': 0,
        'motorcycle': 1,
        'motorbike': 1,
        'car': 2,
        'person': 3,
        'pedestrian': 3,
        'truck': 2,  # truck归类为car
        'bus': 2,    # bus也归类为car
    }
    return class_mapping.get(class_name.lower(), None)


def convert_xml_to_yolo(xml_file, img_width, img_height):
    """解析XML文件并转换为YOLO格式"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片尺寸
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
        return []


def convert_json_to_yolo(json_file, img_width, img_height):
    """解析JSON文件并转换为YOLO格式（如果需要）"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 这里需要根据实际的JSON格式进行解析
        # 暂时返回空列表，如果确实有JSON标签需要转换再补充
        return []
    except:
        return []


def write_yolo_label(annotations, output_path):
    """写入YOLO格式的标签文件"""
    with open(output_path, 'w') as f:
        for class_id, x_center, y_center, w, h in annotations:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def get_image_dimensions(img_path):
    """获取图片尺寸（使用PIL处理中文路径）"""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except:
        return None, None


def process_weather_dataset(weather_type, source_dir, target_base, prefix):
    """
    处理单个天气数据集
    
    Args:
        weather_type: 天气类型 ('rain' 或 'snow')
        source_dir: 源数据目录
        target_base: 目标基础目录
        prefix: 文件名前缀
    """
    print(f"\n{'='*70}")
    print(f"处理 {weather_type} 天气数据集")
    print(f"{'='*70}")
    
    source = Path(source_dir)
    target = Path(target_base) / weather_type
    images_src = source / "images"
    labels_src = source / "labels"
    
    # 创建目标目录
    images_dst = target / "images"
    labels_dst = target / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_files = sorted([
        f for f in images_src.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
    ])
    
    # 获取所有XML标签文件
    xml_files = sorted(labels_src.glob("*.xml"))
    
    print(f"图片数量: {len(image_files)}")
    print(f"XML标签数量: {len(xml_files)}")
    
    # 建立文件名映射
    # 对于雨天：图片是中文名，标签是 "01 (1).xml" 格式
    # 对于雪天：图片和标签都是数字名，但有些特殊字符
    
    success_count = 0
    processed_pairs = []
    
    # 尝试匹配图片和标签
    for i, img_file in enumerate(image_files):
        # 从XML的filename字段找到对应的标签
        matched_xml = None
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename_elem = root.find('filename')
                if filename_elem is not None:
                    xml_filename = filename_elem.text
                    # 检查是否匹配当前图片
                    if xml_filename == img_file.name:
                        matched_xml = xml_file
                        break
            except:
                continue
        
        if matched_xml is None:
            # 如果没找到，尝试通过序号匹配（雪天数据集）
            # 提取图片文件名的数字部分
            img_stem = img_file.stem
            # 处理特殊情况如 "17.jpg.jpg" -> "17.jpg", "29..jpg" -> "29."
            clean_stem = img_stem.split('.')[0]
            
            for xml_file in xml_files:
                xml_stem = xml_file.stem
                # 处理特殊情况如 "17.jpg.xml" -> "17.jpg", "66、.xml" -> "66、"
                xml_clean = xml_stem.split('.')[0]
                
                if clean_stem == xml_clean or img_stem == xml_stem:
                    matched_xml = xml_file
                    break
        
        if matched_xml is not None:
            try:
                # 新文件名
                new_name = f"{prefix}_{success_count+1:04d}"
                new_img_name = f"{new_name}{img_file.suffix}"
                new_lbl_name = f"{new_name}.txt"
                
                # 复制并重命名图片
                shutil.copy2(img_file, images_dst / new_img_name)
                
                # 获取图片尺寸
                img_width, img_height = get_image_dimensions(images_dst / new_img_name)
                
                if img_width and img_height:
                    # 转换XML为YOLO格式
                    annotations = convert_xml_to_yolo(matched_xml, img_width, img_height)
                    
                    if annotations:
                        # 写入YOLO标签
                        write_yolo_label(annotations, labels_dst / new_lbl_name)
                        success_count += 1
                        
                        if success_count % 50 == 0:
                            print(f"  已处理: {success_count} 个样本")
                    else:
                        print(f"  ⚠️  {img_file.name}: 无有效标注")
                else:
                    print(f"  ❌ {img_file.name}: 无法读取图片")
                    
            except Exception as e:
                print(f"  ❌ 处理 {img_file.name} 时出错: {e}")
    
    print(f"\n✅ {weather_type} 处理完成: {success_count} 个样本")
    
    # 统计
    final_images = len(list(images_dst.glob("*")))
    final_labels = len(list(labels_dst.glob("*.txt")))
    print(f"  最终图片: {final_images} 张")
    print(f"  最终标签: {final_labels} 个")
    
    return success_count


def main():
    """主函数"""
    print("="*70)
    print("多天气数据集处理工具")
    print("="*70)
    
    # 配置
    _BASE_DIR = Path(__file__).parent
    target_base = str(_BASE_DIR.parent / "multi_weather_dataset")

    datasets = [
        {
            'type': 'rain',
            'source': r"（需替换为你的原始雨天数据路径）",
            'prefix': 'rain'
        },
        {
            'type': 'snow',
            'source': r"（需替换为你的原始雪天数据路径）",
            'prefix': 'snow'
        }
    ]
    
    total_count = 0
    
    for dataset in datasets:
        count = process_weather_dataset(
            dataset['type'],
            dataset['source'],
            target_base,
            dataset['prefix']
        )
        total_count += count
    
    print("\n" + "="*70)
    print(f"✅ 所有数据集处理完成！")
    print(f"总共处理: {total_count} 个样本")
    print(f"目标目录: {target_base}")
    print("="*70)
    
    # 显示最终结构
    print("\n📁 数据集结构:")
    base = Path(target_base)
    for weather in ['fog', 'rain', 'snow']:
        weather_dir = base / weather
        if weather_dir.exists():
            imgs = len(list((weather_dir / 'images').glob('*')))
            lbls = len(list((weather_dir / 'labels').glob('*.txt')))
            print(f"  {weather}: {imgs} 张图片, {lbls} 个标签")


if __name__ == "__main__":
    main()
