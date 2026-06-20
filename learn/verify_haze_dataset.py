from pathlib import Path

base = Path(__file__).parent.parent / "multi_weather_dataset" / "haze"
images_dir = base / "images"
labels_dir = base / "labels"

imgs = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
lbls = len(list(labels_dir.glob("*.txt")))

print("=" * 60)
print("雾霾数据集验证")
print("=" * 60)
print(f"图片: {imgs} 张")
print(f"标签: {lbls} 个")
print(f"匹配: {'✅ 完全匹配' if imgs == lbls else '❌ 不匹配'}")
print("=" * 60)

# 显示示例
print("\n示例文件:")
sample_imgs = sorted([f.name for f in images_dir.glob("*.jpg")])[:3]
for img_name in sample_imgs:
    lbl_path = labels_dir / img_name.replace('.jpg', '.txt')
    print(f"  图片: {img_name}")
    if lbl_path.exists():
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        print(f"  标签: {lbl_path.name} ({len(lines)} 个标注)")
        if lines:
            print(f"  内容: {lines[0].strip()}")
    print()
