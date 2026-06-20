"""
分析雨天数据集的图片和标签匹配情况
"""
import xml.etree.ElementTree as ET
from pathlib import Path


def analyze_rain_dataset():
    """分析雨天数据集"""
    rain_dir = Path(__file__).parent.parent / "multi_weather_dataset" / "rain"
    images_dir = rain_dir / "images"
    labels_dir = rain_dir / "labels"
    
    # 获取所有图片文件
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    # 获取所有XML文件
    xml_files = sorted(labels_dir.glob("*.xml"))
    
    print("="*80)
    print("雨天数据集匹配分析")
    print("="*80)
    print(f"\n图片数量: {len(image_files)}")
    print(f"XML标签数量: {len(xml_files)}")
    
    # 提取XML中的filename
    xml_filenames = {}
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_elem = root.find('filename')
            if filename_elem is not None:
                xml_filename = filename_elem.text
                xml_filenames[xml_file.name] = xml_filename
        except Exception as e:
            print(f"解析 {xml_file.name} 失败: {e}")
    
    print(f"\n成功解析的XML: {len(xml_filenames)} 个")
    
    # 检查匹配情况
    image_names = set([f.name for f in image_files])
    xml_target_names = set(xml_filenames.values())
    
    matched = image_names & xml_target_names
    in_xml_not_in_images = xml_target_names - image_names
    in_images_not_in_xml = image_names - xml_target_names
    
    print(f"\n【匹配统计】")
    print(f"  完全匹配的: {len(matched)} 个")
    print(f"  XML中有但图片缺失的: {len(in_xml_not_in_images)} 个")
    print(f"  图片有但XML没有的: {len(in_images_not_in_xml)} 个")
    
    if matched:
        print(f"\n✅ 找到 {len(matched)} 个匹配的样本！")
        print("\n匹配示例（前10个）:")
        for i, name in enumerate(sorted(matched)[:10], 1):
            # 找到对应的XML文件
            matching_xml = None
            for xml_name, target_name in xml_filenames.items():
                if target_name == name:
                    matching_xml = xml_name
                    break
            print(f"  {i}. 图片: {name}")
            if matching_xml:
                print(f"     标签: {matching_xml}")
    
    if in_xml_not_in_images:
        print(f"\n⚠️  XML中引用但找不到的图片（前10个）:")
        for i, name in enumerate(sorted(in_xml_not_in_images)[:10], 1):
            print(f"  {i}. {name}")
    
    if in_images_not_in_xml:
        print(f"\n⚠️  没有对应XML的图片（前10个）:")
        for i, name in enumerate(sorted(in_images_not_in_xml)[:10], 1):
            print(f"  {i}. {name}")
    
    # 检查文件名模式
    print(f"\n【文件名模式分析】")
    print("\n图片文件名示例:")
    for img in image_files[:5]:
        print(f"  {img.name}")
    
    print("\nXML目标文件名示例:")
    for target_name in sorted(xml_target_names)[:5]:
        print(f"  {target_name}")
    
    # 尝试找出规律
    print(f"\n【可能的解决方案】")
    if len(in_xml_not_in_images) > 0 and len(in_images_not_in_xml) > 0:
        print("  看起来图片和标签来自不同的批次")
        print("  建议:")
        print("  1. 检查是否有更多的雨天图片未复制")
        print("  2. 或者只使用匹配的 {len(matched)} 个样本")
    
    if matched:
        print(f"\n✅ 可以直接使用这 {len(matched)} 个匹配的样本进行训练！")
    
    print("\n" + "="*80)
    
    return matched, xml_filenames


if __name__ == "__main__":
    analyze_rain_dataset()
