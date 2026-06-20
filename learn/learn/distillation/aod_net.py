"""
AOD-Net (All-in-One Dehazing Network) — IRT 不变重建教师
论文: AOD-Net: All-in-One Dehazing Network (ICCV 2017)
公式: J(x) = K(x) * I(x) - K(x) + b
   其中 K(x) 为 AOD-Net 估计的大气光与透射率融合参数

该模块:
  1. 前向推理输出复原清晰图像
  2. 通过 Sobel 算子提取底层轮廓特征，用于与学生网络做 MSE 蒸馏
  3. 全程冻结，不参与反向传播
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# AOD-Net 模型定义
# ==============================================================================
class AODNet(nn.Module):
    """
    AOD-Net 轻量去雾网络
    参数总量约 1.7K，适合作为教师网络快速推理
    """

    def __init__(self):
        super().__init__()
        # 多尺度特征提取
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=True)
        # 融合卷积，输出 K-estimation map
        self.conv4 = nn.Conv2d(9, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # 可学习的全局偏置 b (公式中的 b)
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Args:
            x: 输入恶劣天气图像 [B, 3, H, W], 值域 [0, 1]
        Returns:
            restored: 复原后的清晰图像 [B, 3, H, W]
            k_map:    K-estimation 参数图 [B, 3, H, W]
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        cat = torch.cat([x1, x2, x3], dim=1)  # [B, 9, H, W]
        k = self.conv4(cat)                     # K-estimation map
        # J = K * I - K + b
        restored = k * x - k + self.b
        restored = torch.clamp(restored, 0.0, 1.0)
        return restored, k

    def load_pretrained(self, weights_path):
        """加载预训练权重"""
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)
        print(f"[IRT] AOD-Net 预训练权重已加载: {weights_path}")


# ==============================================================================
# Sobel 边缘特征提取器 — 用于 IRT 蒸馏损失
# ==============================================================================
class SobelEdgeExtractor(nn.Module):
    """
    基于 Sobel 算子的边缘/轮廓特征提取
    从 AOD-Net 复原图像中提取底层结构特征
    输出多尺度边缘特征图，与学生网络 Neck 特征对齐
    """

    def __init__(self, output_channels=256):
        super().__init__()
        # Sobel 核 (固定，不参与训练)
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        # 扩展到 3 通道
        sobel_x = sobel_x.repeat(3, 1, 1, 1)  # [3, 1, 3, 3]
        sobel_y = sobel_y.repeat(3, 1, 1, 1)  # [3, 1, 3, 3]
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 边缘特征到蒸馏空间的投影
        self.edge_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def extract_edges(self, image):
        """
        从图像提取 Sobel 边缘幅度图
        Args:
            image: [B, 3, H, W], 值域 [0, 1]
        Returns:
            edge_mag: [B, 3, H, W]
        """
        grad_x = F.conv2d(image, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(image, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def forward(self, restored_image):
        """
        Args:
            restored_image: AOD-Net 输出的复原图像 [B, 3, H, W]
        Returns:
            edge_features: 用于蒸馏的底层轮廓特征 [B, out_c, H/2, W/2]
        """
        edges = self.extract_edges(restored_image)         # [B, 3, H, W]
        edge_features = self.edge_proj(edges)               # [B, out_c, H/2, W/2]
        return edge_features


# ==============================================================================
# IRT 教师完整封装
# ==============================================================================
class IRTTeacher(nn.Module):
    """
    IRT 不变重建教师
    组合 AOD-Net + SobelEdgeExtractor，全程冻结

    使用方式:
        teacher = IRTTeacher(output_channels=256)
        teacher.load_aod_weights("aod_net.pth")  # 可选
        teacher.freeze()
        restored_img, edge_feat = teacher(weather_image)
    """

    def __init__(self, output_channels=256):
        super().__init__()
        self.aod_net = AODNet()
        self.edge_extractor = SobelEdgeExtractor(output_channels=output_channels)

    def load_aod_weights(self, weights_path):
        self.aod_net.load_pretrained(weights_path)

    def freeze(self):
        """冻结所有参数，关闭梯度"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        print("[IRT] IRT 教师已冻结，参数数量:", sum(p.numel() for p in self.parameters()))

    def forward(self, weather_image):
        """
        Args:
            weather_image: 恶劣天气原图 [B, 3, H, W], 值域 [0, 1]
        Returns:
            restored:    复原后的清晰图像 [B, 3, H, W]
            edge_feat:   底层轮廓特征 [B, out_c, H/2, W/2]
        """
        with torch.no_grad():
            restored, k_map = self.aod_net(weather_image)
            edge_feat = self.edge_extractor(restored)
        return restored, edge_feat


# ==============================================================================
# 预训练权重下载 / 初始化
# ==============================================================================
def create_irt_teacher(output_channels=256, pretrained_path=None):
    """
    工厂函数：创建 IRT 教师并加载预训练权重

    如果没有预训练权重，AOD-Net 将使用随机初始化。
    由于 teacher 作用是通过 Sobel 边缘蒸馏底层结构信息，
    AOD-Net 未预训练时仍有边缘提取能力，但去雾效果会差。

    推荐使用预训练权重，可从以下渠道获取:
    - https://github.com/weichen582/AOD-Net
    - 或自行在 RESIDE/OTS 数据集上预训练
    """
    teacher = IRTTeacher(output_channels=output_channels)
    if pretrained_path is not None and os.path.exists(pretrained_path):
        teacher.load_aod_weights(pretrained_path)
    else:
        print("[IRT] 未找到预训练权重，使用随机初始化。"
              "建议下载 AOD-Net 预训练模型以获得更好的去雾效果。")
    teacher.freeze()
    return teacher


# 测试入口
if __name__ == "__main__":
    import os
    dummy = torch.randn(2, 3, 640, 640).clamp(0, 1)
    teacher = IRTTeacher(output_channels=256)
    teacher.freeze()
    restored, edge = teacher(dummy)
    print(f"输入:   {dummy.shape}")        # [2, 3, 640, 640]
    print(f"复原图: {restored.shape}")      # [2, 3, 640, 640]
    print(f"边缘特征: {edge.shape}")        # [2, 256, 320, 320]
    print(f"总参数:  {sum(p.numel() for p in teacher.parameters()):,}")
