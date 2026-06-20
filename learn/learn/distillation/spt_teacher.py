"""
SPT 语义先验教师 — 基于预训练雾天 YOLO 模型
全程冻结，仅做前向推理，输出 Neck 层中间语义特征用于蒸馏

特征提取层级:
  - P3 层 (layer 15): 高分辨率浅层特征 [B, 64, 80, 80]
  - P4 层 (layer 18/22): 中分辨率特征 [B, 128, 40, 40]
  - P5 层 (layer 21/25): 低分辨率深层语义 [B, 256, 20, 20]
"""
import torch
import torch.nn as nn
import sys
import os


class SPTTeacher(nn.Module):
    """
    SPT 语义先验教师

    加载预训练的 YOLO-FCA 检测模型权重 (best.pt)，
    提取 Neck 层特征图用于语义蒸馏。
    全程冻结，关闭梯度回传。
    """

    def __init__(self, weights_path, device="cuda:0"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 从 ultralytics 加载完整模型
        from ultralytics import YOLO
        self.yolo = YOLO(weights_path)
        self.model = self.yolo.model  # DetectionModel
        self.model.to(self.device)

        # 注册特征提取钩子
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        """在关键 Neck 层注册前向钩子，捕获中间特征"""
        # YOLOv8-FCA neck 结构中的关键层索引
        # 15: P3/8-small (上采样后融合)  → 高分辨率
        # 19: P3/8-small (bottom-up后)   → 中分辨率
        # 22: P4/16-medium               → 中分辨率
        # 25: P5/32-large                → 低分辨率
        self.hook_layers = [15, 22, 25]
        self.hook_handles = []

        for idx in self.hook_layers:
            if idx < len(self.model.model):
                handle = self.model.model[idx].register_forward_hook(
                    self._make_hook(idx)
                )
                self.hook_handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook

    def freeze(self):
        """冻结 SPT 教师所有参数"""
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[SPT] SPT 教师已冻结，总参数: {total:,}")

    def forward(self, x):
        """
        前向传播获取 Neck 语义特征

        Args:
            x: 输入图像 [B, 3, H, W], 值域 [0, 1]
        Returns:
            features: dict {layer_idx: tensor}
                - 15: P3 高分辨率特征 [B, 64, H/8, W/8]
                - 22: P4 中分辨率特征 [B, 128, H/16, W/16]
                - 25: P5 低分辨率特征 [B, 256, H/32, W/32]
        """
        with torch.no_grad():
            self.features = {}
            _ = self.model(x)
            # 返回深拷贝，避免外部修改影响内部状态
            return {k: v.clone() for k, v in self.features.items() if v is not None}

    def remove_hooks(self):
        """清理钩子"""
        for h in self.hook_handles:
            h.remove()

    def get_neck_features(self, x):
        """
        便捷方法：获取标准化后的 Neck 特征列表
        用于蒸馏损失计算

        Returns:
            feat_list: List[Tensor], P3→P5 特征
        """
        feats = self.forward(x)
        # 按层级顺序返回
        return [feats[idx] for idx in sorted(feats.keys())]


def create_spt_teacher(weights_path, device="cuda:0"):
    """
    工厂函数：创建 SPT 教师

    Args:
        weights_path: 预训练 YOLO-FCA 权重路径 (best.pt)
                      如果不存在，将抛出 FileNotFoundError
        device: 设备

    Returns:
        SPTTeacher 实例，已冻结
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[SPT] 预训练权重未找到: {weights_path}\n"
            f"请先训练一个雾天 YOLO-FCA 模型获取 best.pt，\n"
            f"或指定已有模型权重路径。"
        )
    teacher = SPTTeacher(weights_path, device=device)
    teacher.freeze()
    return teacher


# 测试入口
if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_BASE_DIR, "..", "ultralytics-main"))

    # 使用已有的训练权重
    weights = os.path.join(_BASE_DIR, "..", "runs", "fca_detect",
                           "train_fca_cl", "weights", "best.pt")

    if os.path.exists(weights):
        spt = SPTTeacher(weights, device="cuda")
        spt.freeze()
        dummy = torch.randn(2, 3, 640, 640).to(spt.device)
        feats = spt(dummy)
        for k, v in feats.items():
            print(f"Layer {k}: {v.shape}")
        spt.remove_hooks()
    else:
        print(f"[SPT] 权重文件不存在: {weights}")
        print("请先运行 train.py 训练一个基础模型。")
