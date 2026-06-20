"""
数据集准备：将 multi_weather_dataset 按 8:1:1 拆分为 train/val/test
统一 fog/haze/rain/snow 四类恶劣天气数据
"""
import os
import random
from pathlib import Path
import shutil

random.seed(42)

_BASE_DIR = Path(__file__).parent.parent.parent.parent  # learn (2)/
_SRC = _BASE_DIR / "multi_weather_dataset"
_DST = _BASE_DIR / "adverse_weather_yolo"

# 类别映射保持一致
CLASS_NAMES = {0: "自行车", 1: "摩托车", 2: "汽车", 3: "人"}
NC = 4


def collect_pairs(weather_type):
    """收集单个天气类型下的所有 图像-标签 对"""
    img_dir = _SRC / weather_type / "images"
    lbl_dir = _SRC / weather_type / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return []
    pairs = []
    for img_path in img_dir.glob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


def main():
    all_pairs = []
    for wt in ["fog", "haze", "rain", "snow"]:
        pairs = collect_pairs(wt)
        print(f"{wt}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    print(f"Total: {len(all_pairs)} pairs")

    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    for split_name, pairs in splits.items():
        img_dst = _DST / "images" / split_name
        lbl_dst = _DST / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
        print(f"{split_name}: {len(pairs)} pairs -> {img_dst}")

    print(f"\n✅ 数据集已生成: {_DST}")
    print("目录结构: images/train, images/val, images/test, labels/train, labels/val, labels/test")


if __name__ == "__main__":
    main()
