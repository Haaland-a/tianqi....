
"""
FCA-CL Module: Foggy weather small target perception
Cross-layer feature Aggregation and Contrastive Learning enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossLayerFeatureAggregation(nn.Module):
    """
    跨层特征聚合模块
    融合 P2 浅层高分辨率特征和 P3 深层语义特征
    公式: F_small = α · P2_conv + β · Up(P3)  (α + β = 1)
    """

    def __init__(self, p2_channels, p3_channels, small_target_channels=128):
        super().__init__()

        # P2 特征转换
        self.p2_conv = nn.Sequential(
            nn.Conv2d(p2_channels, small_target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(small_target_channels),
            nn.SiLU(inplace=True)
        )

        # P3 上采样和转换
        self.p3_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(p3_channels, small_target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(small_target_channels),
            nn.SiLU(inplace=True)
        )

        # 可学习的注意力权重（自适应融合）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # 小目标特征增强
        self.small_target_enhance = nn.Sequential(
            nn.Conv2d(small_target_channels, small_target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(small_target_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(small_target_channels, small_target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(small_target_channels),
        )

    def forward(self, p2, p3):
        """
        Args:
            p2: P2层特征 [B, C2, H/4, W/4]
            p3: P3层特征 [B, C3, H/8, W/8]
        Returns:
            融合后的小目标特征 [B, small_target_channels, H/4, W/4]
        """
        # P2 特征转换
        p2_feat = self.p2_conv(p2)

        # P3 特征上采样和转换
        p3_feat = self.p3_upsample(p3)

        # 归一化注意力权重（确保 α + β = 1）
        alpha_weight = torch.sigmoid(self.alpha)
        beta_weight = 1 - alpha_weight

        # 跨层特征聚合
        fused_feat = alpha_weight * p2_feat + beta_weight * p3_feat

        # 小目标特征增强
        enhanced_feat = self.small_target_enhance(fused_feat) + fused_feat

        return enhanced_feat


class ContrastiveLearningEnhancement(nn.Module):
    """
    对比学习增强模块
    使用 InfoNCE 损失函数拉近同类小目标特征距离，推远与雾噪声特征距离
    """

    def __init__(self, feature_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
        """
        Args:
            features: 特征图 [B, C, H, W]
        Returns:
            投影后的特征 [B, H*W, feature_dim]
        """
        B, C, H, W = features.shape
        # 展平空间维度
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # 投影到低维空间
        projected = self.projection(features_flat)
        return F.normalize(projected, dim=-1)

    @staticmethod
    def info_nce_loss(query, positive, negatives, temperature=0.07):
        """
        InfoNCE 损失函数
        公式: L_cont = -log(exp(sim(q, k+)/τ) / (exp(sim(q, k+)/τ) + Σexp(sim(q, k-)/τ)))

        Args:
            query: 查询特征 [B, N, D]
            positive: 正样本特征 [B, N, D]
            negatives: 负样本特征 [B, M, D]
            temperature: 温度参数
        Returns:
            对比学习损失
        """
        # 计算查询与正样本的相似度
        pos_sim = F.cosine_similarity(query, positive, dim=-1) / temperature  # [B, N]

        # 计算查询与负样本的相似度
        neg_sim = torch.einsum('bnd,bmd->bnm', query, negatives) / temperature  # [B, N, M]

        # InfoNCE 损失
        pos_exp = torch.exp(pos_sim)  # [B, N]
        neg_exp = torch.exp(neg_sim).sum(dim=-1)  # [B, N]

        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))

        return loss.mean()


class FCACLModule(nn.Module):
    """
    完整的 FCA-CL 模块
    融合跨层特征聚合和对比学习增强
    """

    def __init__(self, p2_channels, p3_channels, small_target_channels=128):
        super().__init__()

        # 跨层特征聚合
        self.feature_aggregation = CrossLayerFeatureAggregation(
            p2_channels=p2_channels,
            p3_channels=p3_channels,
            small_target_channels=small_target_channels
        )

        # 对比学习增强
        self.contrastive_learning = ContrastiveLearningEnhancement(
            feature_dim=small_target_channels
        )

        # 输出投影（保持与原始特征相同的通道数）
        self.output_proj = nn.Conv2d(small_target_channels, p2_channels, kernel_size=1, bias=False)

    def forward(self, p2, p3, use_contrastive=True):
        """
        Args:
            p2: P2层特征 [B, C2, H/4, W/4]
            p3: P3层特征 [B, C3, H/8, W/8]
            use_contrastive: 是否使用对比学习（训练时为True，推理时为False）
        Returns:
            增强后的P2特征, 对比学习损失（如果需要）
        """
        # 跨层特征聚合
        aggregated_feat = self.feature_aggregation(p2, p3)

        if use_contrastive and self.training:
            # 对比学习增强
            contrastive_feat = self.contrastive_learning(aggregated_feat)

            # 将聚合后的特征投影回原始通道数
            output_feat = self.output_proj(aggregated_feat)

            return output_feat, contrastive_feat
        else:
            output_feat = self.output_proj(aggregated_feat)
            return output_feat, None


# 为了与 YOLOv8 兼容，创建一个简单的包装类
class FCA(nn.Module):
    """
    简化版 FCA 模块（用于 YOLO YAML 配置）
    仅包含跨层特征聚合，对比学习在损失函数中实现
    """

    def __init__(self, c1, c2, c3=None):
        """
        Args:
            c1: P2 通道数
            c2: P3 通道数
        """
        super().__init__()
        if c3 is None:
            c3 = c1  # 默认输出通道与 P2 相同

        self.p2_conv = nn.Sequential(
            nn.Conv2d(c1, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True)
        )

        self.p3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True)
        )

        # 可学习权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.enhance = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
        )

    def forward(self, x):
        """
        期望输入为元组: (p2_features, p3_features)
        """
        if isinstance(x, tuple) or isinstance(x, list):
            p2, p3 = x
        else:
            # 如果只有一个输入，可能是 Concat 之后的结果，需要拆分
            raise ValueError("FCA module expects tuple input (p2, p3)")

        p2_feat = self.p2_conv(p2)
        p3_feat = self.p3_up(p3)

        alpha = torch.sigmoid(self.alpha)
        beta = 1 - alpha

        fused = alpha * p2_feat + beta * p3_feat
        out = self.enhance(fused) + fused

        return out
