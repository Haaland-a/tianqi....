"""
双教师知识蒸馏损失函数

总损失公式:
    Total_Loss = Det_Loss + λ1 * IRT_Distill_Loss + λ2 * SPT_Distill_Loss

其中:
    Det_Loss          — YOLO 原生检测损失 (box + cls + dfl)
    IRT_Distill_Loss  — 学生 Neck 特征与 IRT 边缘特征的 MSE
    SPT_Distill_Loss  — 学生 Neck 特征与 SPT 语义特征的 MSE

权重配比说明 (λ1, λ2):
  - 默认 λ1=0.3, λ2=1.0
  - λ2 > λ1: 语义蒸馏比边缘蒸馏更重要，因为高层语义对检测帮助更大
  - 调优建议:
    - 如果学生检测精度低于预期 → 增大 λ2 (加强语义蒸馏)
    - 如果学生去雾后检测不稳定 → 增大 λ1 (加强重建蒸馏)
    - 如果训练初期损失震荡 → 降低 λ1, λ2 或用 warmup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    """
    特征对齐损失 — 自适应将学生特征对齐到教师特征空间
    使用可学习的投影层，避免维度不匹配导致的蒸馏失败
    """

    def __init__(self, student_ch, teacher_ch, aligned_ch=128):
        super().__init__()
        self.student_proj = nn.Sequential(
            nn.Conv2d(student_ch, aligned_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aligned_ch),
            nn.ReLU(inplace=True),
        )
        self.teacher_proj = nn.Sequential(
            nn.Conv2d(teacher_ch, aligned_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aligned_ch),
            nn.ReLU(inplace=True),
        )
        # 注意: 投影层参数可以训练，帮助特征对齐

    def forward(self, student_feat, teacher_feat):
        s = self.student_proj(student_feat)
        t = self.teacher_proj(teacher_feat)
        return F.mse_loss(s, t)


class IRTDistillLoss(nn.Module):
    """
    IRT 蒸馏损失: 学生特征与 IRT 提取的底层轮廓特征做 MSE

    边缘特征 (IRT) 来自 AOD-Net 复原图像 + Sobel 边缘提取
    学生 Neck 特征需要与这些底层结构特征对齐

    实际使用:
        提取学生网络 Neck 层的某一级特征 (如 P3 高分辨率层)
        与 IRT 边缘特征做自适应对齐 + MSE
    """

    def __init__(self, student_ch=64, irt_ch=256, aligned_ch=128):
        super().__init__()
        self.alignment = FeatureAlignmentLoss(student_ch, irt_ch, aligned_ch)

    def forward(self, student_feat, irt_edge_feat):
        """
        Args:
            student_feat:   学生 Neck 特征 [B, student_ch, H, W]
            irt_edge_feat:  IRT 边缘特征 [B, irt_ch, H_edge, W_edge]
        Returns:
            loss: 标量
        """
        # 尺寸对齐
        if student_feat.shape[2:] != irt_edge_feat.shape[2:]:
            irt_edge_feat = F.interpolate(
                irt_edge_feat, size=student_feat.shape[2:], mode="bilinear", align_corners=False
            )
        return self.alignment(student_feat, irt_edge_feat)


class SPTDistillLoss(nn.Module):
    """
    SPT 蒸馏损失: 学生 Neck 特征与 SPT 提取的语义特征做 MSE

    SPT 提供三层语义特征 (P3/P4/P5)，学生需要逐层对齐
    采用多尺度加权: P3(高分辨率)权重略小，P5(深层语义)权重略大
    """

    def __init__(self, student_channels=None, spt_channels=None, aligned_ch=128):
        super().__init__()
        if student_channels is None:
            student_channels = [64, 128, 256]   # P3, P4, P5
        if spt_channels is None:
            spt_channels = [64, 128, 256]       # P3, P4, P5

        self.alignments = nn.ModuleList([
            FeatureAlignmentLoss(sc, tc, aligned_ch)
            for sc, tc in zip(student_channels, spt_channels)
        ])
        # 多尺度权重: P5 深层语义权重最高
        self.w_p3 = nn.Parameter(torch.tensor(0.3))
        self.w_p4 = nn.Parameter(torch.tensor(0.3))
        self.w_p5 = nn.Parameter(torch.tensor(0.4))

    def forward(self, student_feats, spt_feats):
        """
        Args:
            student_feats: List[Tensor], 学生 P3/P4/P5 Neck 特征
            spt_feats:     List[Tensor], SPT  P3/P4/P5 Neck 特征
        Returns:
            loss: 加权多尺度 SPT 蒸馏损失
        """
        assert len(student_feats) == len(spt_feats) == len(self.alignments), \
            f"特征层数不匹配: {len(student_feats)} vs {len(self.alignments)}"

        losses = []
        for i, (sf, tf) in enumerate(zip(student_feats, spt_feats)):
            if sf.shape[2:] != tf.shape[2:]:
                tf = F.interpolate(tf, size=sf.shape[2:], mode="bilinear", align_corners=False)
            loss_i = self.alignments[i](sf, tf)
            w = [self.w_p3, self.w_p4, self.w_p5][i]
            losses.append(loss_i * w)

        total = sum(losses) / sum([self.w_p3, self.w_p4, self.w_p5])
        return total


class DualTeacherDistillLoss(nn.Module):
    """
    双教师蒸馏总损失

    总损失 = Det_Loss + λ1 * IRT_Distill_Loss + λ2 * SPT_Distill_Loss

    Args:
        lambda_irt: IRT 蒸馏权重, 默认 0.3
        lambda_spt: SPT 蒸馏权重, 默认 1.0
        student_channels: 学生 Neck 各层通道
        spt_channels:     SPT Neck 各层通道
        irt_channels:     IRT 边缘特征通道
    """

    def __init__(self, lambda_irt=0.3, lambda_spt=1.0,
                 student_channels=None, spt_channels=None, irt_channels=256,
                 aligned_ch=128):
        super().__init__()
        if student_channels is None:
            student_channels = [64, 128, 256]
        if spt_channels is None:
            spt_channels = [64, 128, 256]

        self.lambda_irt = lambda_irt
        self.lambda_spt = lambda_spt

        self.irt_loss = IRTDistillLoss(
            student_ch=student_channels[0],  # P3 (最高分辨率) 对齐 IRT 边缘
            irt_ch=irt_channels,
            aligned_ch=aligned_ch,
        )
        self.spt_loss = SPTDistillLoss(
            student_channels=student_channels,
            spt_channels=spt_channels,
            aligned_ch=aligned_ch,
        )

    def forward(self, det_loss, student_neck_feats, irt_edge_feat, spt_neck_feats):
        """
        Args:
            det_loss:            YOLO 原生检测损失 (标量)
            student_neck_feats:  List[Tensor], 学生 Neck 特征 [P3, P4, P5]
            irt_edge_feat:       IRT 边缘特征 Tensor
            spt_neck_feats:      List[Tensor], SPT Neck 特征 [P3, P4, P5]
        Returns:
            total_loss:         组合总损失
            loss_dict:          dict, 各分量损失值 (用于日志)
        """
        # IRT 蒸馏 (学生 P3 特征 vs IRT 边缘)
        irt_distill = self.irt_loss(student_neck_feats[0], irt_edge_feat)

        # SPT 蒸馏 (学生 P3/P4/P5 vs SPT)
        spt_distill = self.spt_loss(student_neck_feats, spt_neck_feats)

        # 总损失
        total = det_loss + self.lambda_irt * irt_distill + self.lambda_spt * spt_distill

        loss_dict = {
            "det_loss": det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss,
            "irt_distill": irt_distill.item(),
            "spt_distill": spt_distill.item(),
            "lambda_irt": self.lambda_irt,
            "lambda_spt": self.lambda_spt,
            "total": total.item() if isinstance(total, torch.Tensor) else total,
        }
        return total, loss_dict


# 测试入口
if __name__ == "__main__":
    B = 2
    # 模拟数据
    det_loss = torch.tensor(2.5)
    student_feats = [
        torch.randn(B, 64, 80, 80),
        torch.randn(B, 128, 40, 40),
        torch.randn(B, 256, 20, 20),
    ]
    irt_edge = torch.randn(B, 256, 160, 160)
    spt_feats = [
        torch.randn(B, 64, 80, 80),
        torch.randn(B, 128, 40, 40),
        torch.randn(B, 256, 20, 20),
    ]

    loss_fn = DualTeacherDistillLoss(lambda_irt=0.3, lambda_spt=1.0)
    total, info = loss_fn(det_loss, student_feats, irt_edge, spt_feats)
    print("=" * 50)
    print("双教师蒸馏损失测试")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
