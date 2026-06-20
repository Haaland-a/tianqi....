"""
成对数据加载器 — 恶劣天气原图 + IRT 复原清晰图

每个 batch 返回:
  - weather_img:   原始恶劣天气图像 (归一化到 [0,1])
  - restored_img:  IRT 教师生成的复原清晰图
  - labels:        标准 YOLO 格式标注 (class, cx, cy, w, h)
  - img_path:      原始图像路径 (用于调试)
  - shape:         原始图像尺寸 (h, w)
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class PairedWeatherDataset(Dataset):
    """
    成对恶劣天气数据集

    每次迭代返回原始恶劣天气图像 + 对应标签。
    IRT 复原图像在 collate 或训练循环中批量生成，
    避免预先遍历整个数据集。
    """

    def __init__(self, img_dir, lbl_dir, img_size=640, augment=False,
                 allowed_exts=None):
        """
        Args:
            img_dir:   图片目录 (如 adverse_weather_yolo/images/train/)
            lbl_dir:   标签目录 (如 adverse_weather_yolo/labels/train/)
            img_size:  统一缩放尺寸
            augment:   是否启用数据增强
        """
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.augment = augment

        if allowed_exts is None:
            allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        # 收集所有成对样本
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*")):
            if img_path.suffix.lower() in allowed_exts:
                lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    self.samples.append((img_path, lbl_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"未找到有效样本: img={img_dir}, lbl={lbl_dir}")

        print(f"[Dataset] {Path(img_dir).parent.name}/{Path(img_dir).name}: "
              f"{len(self.samples)} 对样本")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        """加载图像并归一化"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_labels(self, path, img_h, img_w):
        """加载 YOLO 格式标签 (归一化坐标)"""
        labels = []
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h_val = float(parts[4])
                        labels.append([cls_id, cx, cy, w, h_val])
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _resize_and_pad(self, img, labels):
        """resize + letterbox, 保持宽高比"""
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            new_h, new_w = int(h0 * r), int(w0 * r)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # letterbox 填充到 img_size x img_size
        h, w = img.shape[:2]
        dw = self.img_size - w
        dh = self.img_size - h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 调整标签坐标
        if len(labels) > 0:
            labels[:, 1] = (labels[:, 1] * w0 * r + left) / self.img_size
            labels[:, 2] = (labels[:, 2] * h0 * r + top) / self.img_size
            labels[:, 3] = labels[:, 3] * w0 * r / self.img_size
            labels[:, 4] = labels[:, 4] * h0 * r / self.img_size

        return img, labels, (r, left, top)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # 加载
        img_raw = self._load_image(img_path)
        h0, w0 = img_raw.shape[:2]
        labels = self._load_labels(lbl_path, h0, w0)

        # resize + letterbox
        img_resized, labels, (r, left, top) = self._resize_and_pad(img_raw, labels)

        # HWC → CHW, [0,255] → [0,1]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        return {
            "img": img_tensor,                       # [3, 640, 640]
            "labels": torch.from_numpy(labels),      # [N, 5]
            "img_path": str(img_path),
            "shape": (h0, w0),                       # 原始尺寸
            "ratio_pad": (r, left, top),              # resize 参数
        }


def collate_fn(batch):
    """自定义 batch 整理函数，处理变长标签"""
    imgs = torch.stack([item["img"] for item in batch])
    labels = [item["labels"] for item in batch]
    img_paths = [item["img_path"] for item in batch]
    shapes = [item["shape"] for item in batch]
    ratios = [item["ratio_pad"] for item in batch]
    return {
        "img": imgs,
        "labels": labels,
        "img_path": img_paths,
        "shape": shapes,
        "ratio_pad": ratios,
    }


def create_dataloaders(data_root, img_size=640, batch_size=8, workers=4):
    """
    创建 train/val/test DataLoader

    Args:
        data_root: adverse_weather_yolo/ 根目录路径
        img_size:  输入图像尺寸
        batch_size: batch 大小
        workers:   数据加载线程数
    Returns:
        train_loader, val_loader, test_loader
    """
    data_root = Path(data_root)
    loaders = {}
    for split in ["train", "val", "test"]:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        if not img_dir.exists():
            print(f"[Dataset] 跳过 {split} (目录不存在: {img_dir})")
            continue
        is_train = (split == "train")
        ds = PairedWeatherDataset(
            img_dir=img_dir, lbl_dir=lbl_dir,
            img_size=img_size, augment=is_train,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train,
            num_workers=workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=is_train,
        )
        loaders[split] = loader
        print(f"[Dataloader] {split}: {len(ds)} 样本, {len(loader)} batches")

    return loaders.get("train"), loaders.get("val"), loaders.get("test")


# 测试入口
if __name__ == "__main__":
    _BASE_DIR = Path(__file__).parent.parent.parent
    ds_path = _BASE_DIR / "adverse_weather_yolo"

    if not ds_path.exists():
        print(f"数据集目录不存在: {ds_path}")
        print("请先运行 setup_dataset.py 创建数据集")
    else:
        train_l, val_l, test_l = create_dataloaders(ds_path, img_size=640,
                                                     batch_size=4, workers=0)
        if train_l:
            batch = next(iter(train_l))
            print(f"\nBatch 示例:")
            print(f"  img:    {batch['img'].shape}")      # [B, 3, 640, 640]
            print(f"  labels: {len(batch['labels'])} 张图像的标签")
            print(f"  path:   {batch['img_path'][0]}")
