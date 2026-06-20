"""
删除雾霾数据集中没有对应标签的多余图片
"""
from pathlib import Path


def remove_unmatched_images():
    """删除没有对应TXT标签的图片"""
    base_dir = Path(r"D:\数据集\雾霾")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    print("=" * 70)
    print("清理雾霾数据集 - 删除多余图片")
    print("=" * 70)
    
    # 获取所有图片和标签
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    label_files = set([f.stem for f in labels_dir.glob("*.txt")])
    
    print(f"\n📊 当前状态:")
    print(f"  图片总数: {len(image_files)}")
    print(f"  标签总数: {len(label_files)}")
    
    # 找出没有标签的图片
    unmatched = []
    for img_file in image_files:
        if img_file.stem not in label_files:
            unmatched.append(img_file)
    
    print(f"\n⚠️  发现 {len(unmatched)} 个没有标签的图片")
    
    if not unmatched:
        print("\n✅ 所有图片都有对应的标签！")
        return
    
    # 显示未匹配的图片
    print("\n未匹配的图片列表:")
    for i, img in enumerate(unmatched[:10], 1):
        print(f"  {i}. {img.name}")
    if len(unmatched) > 10:
        print(f"  ... 还有 {len(unmatched) - 10} 个")
    
    # 确认删除
    confirm = input(f"\n是否删除这 {len(unmatched)} 个图片？(yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        deleted_count = 0
        for img_file in unmatched:
            try:
                img_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ 删除失败 {img_file.name}: {e}")
        
        print(f"\n✅ 成功删除 {deleted_count} 个图片")
        
        # 最终统计
        final_images = len([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        final_labels = len(list(labels_dir.glob("*.txt")))
        
        print(f"\n📊 最终状态:")
        print(f"  图片: {final_images} 张")
        print(f"  标签: {final_labels} 个")
        
        if final_images == final_labels:
            print("\n✅ 数据集已完全匹配！")
        else:
            print(f"\n⚠️  仍有不匹配，请检查")
    else:
        print("\n❌ 已取消删除操作")
    
    print("=" * 70)


if __name__ == "__main__":
    remove_unmatched_images()
