
"""
DTFA: Dual Teacher Feature Alignment Framework
双教师特征对齐框架 - 针对恶劣天气下的目标检测

核心创新点：
1. IRT教师（Image Reconstruction Teacher）：提供"天气不变的特征先验"
2. SPT教师（Semantic Perception Teacher）：提供"任务感知语义"
3. FCA-CL增强的三个关键位置：
   - 位置一：AFB模块内的显式对比学习对齐
   - 位置二：特征级对比损失
   - 位置三：IRT编码器中的FCA特征聚集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRTTeacher(nn.Module):
    """
    IRT教师网络（Image Reconstruction Teacher）
    
    功能：
    1. 从退化图像中重建干净图像
    2. 提取天气不变的特征表示
    3. 在编码器中集成FCA特征聚集模块进行结构化处理
    
    架构：
    - Encoder: 下采样提取多尺度特征（集成FCA模块）
    - Bottleneck: 特征压缩与增强
    - Decoder: 上采样重建干净图像
    """
    
    def __init__(self, in_channels=3, base_channels=64, use_fca=True):
        super().__init__()
        self.use_fca = use_fca
        
        # === Encoder ===
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # === FCA特征聚集模块（位置三的创新点）===
        if use_fca:
            # 在encoder的conv2和conv3之间插入FCA模块
            self.fca_aggregation = nn.Sequential(
                nn.Conv2d(base_channels * 2 + base_channels * 4, base_channels * 4, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        
        # === Bottleneck ===
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # === Decoder ===
        self.dec_upsample1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # dec_conv1 needs to handle concatenation of dec1 (4C) and enc4_up (8C) = 12C
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(base_channels * 12, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.dec_upsample2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # dec_conv2 needs to handle concatenation of dec2 (2C) and enc3/fca (4C) = 6C
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(base_channels * 6, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.dec_upsample3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2),  # Output 2C to match enc2
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # dec_conv3 needs to handle concatenation of dec3 (2C) and enc2 (2C) = 4C
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: 退化图像 [B, C, H, W]
        Returns:
            reconstructed: 重建的干净图像 [B, C, H, W]
            features: 中间特征字典（用于特征对齐）
        """
        features = {}
        
        # Encoder
        enc1 = self.enc_conv1(x)  # [B, C, H, W]
        enc2 = self.enc_conv2(enc1)  # [B, 2C, H/2, W/2]
        enc3 = self.enc_conv3(enc2)  # [B, 4C, H/4, W/4]
        enc4 = self.enc_conv4(enc3)  # [B, 8C, H/8, W/8]
        
        features['enc1'] = enc1
        features['enc2'] = enc2
        features['enc3'] = enc3
        features['enc4'] = enc4
        
        # FCA特征聚集（位置三的创新点）
        if self.use_fca:
            # 上采样enc3到enc2的尺寸
            enc3_up = F.interpolate(enc3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
            # 拼接enc2和上采样的enc3
            fused = torch.cat([enc2, enc3_up], dim=1)
            # FCA聚集
            fca_feat = self.fca_aggregation(fused)
            features['fca_enhanced'] = fca_feat
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        features['bottleneck'] = bottleneck
        
        # Decoder with skip connections
        dec1 = self.dec_upsample1(bottleneck)  # [B, 4C, H/4, W/4]
        # enc4 is [B, 8C, H/8, W/8], need to upsample to match dec1
        enc4_up = F.interpolate(enc4, size=dec1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc4_up], dim=1)  # [B, 4C+8C=12C, H/4, W/4]
        dec1 = self.dec_conv1(dec1)
        
        dec2 = self.dec_upsample2(dec1)  # [B, 2C, H/2, W/2]
        # 如果使用了FCA，则用增强后的特征替换enc3
        if self.use_fca and 'fca_enhanced' in features:
            # fca_feat is [B, 4C, H/2, W/2], need to ensure it matches dec2
            fca_feat_aligned = features['fca_enhanced']
            if fca_feat_aligned.shape[2:] != dec2.shape[2:]:
                fca_feat_aligned = F.interpolate(fca_feat_aligned, size=dec2.shape[2:], mode='bilinear', align_corners=False)
            dec2 = torch.cat([dec2, fca_feat_aligned], dim=1)
        else:
            # enc3 is [B, 4C, H/4, W/4], need to upsample to match dec2
            enc3_up = F.interpolate(enc3, size=dec2.shape[2:], mode='bilinear', align_corners=False)
            dec2 = torch.cat([dec2, enc3_up], dim=1)
        dec2 = self.dec_conv2(dec2)
        
        dec3 = self.dec_upsample3(dec2)  # [B, 2C, H, W]
        # enc2 is [B, 2C, H/2, W/2], need to upsample to match dec3
        enc2_up = F.interpolate(enc2, size=dec3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc2_up], dim=1)  # [B, 2C+2C=4C, H, W]
        dec3 = self.dec_conv3(dec3)
        
        # Output
        reconstructed = self.output(dec3)
        features['reconstructed'] = reconstructed
        
        return reconstructed, features


class SPTTeacher(nn.Module):
    """
    SPT教师网络（Semantic Perception Teacher）
    
    功能：
    1. 在干净图像上预训练的标准检测器
    2. 提供高质量的任务感知语义特征
    3. 指导学生网络的检测头学习
    
    架构：
    - 使用标准的YOLOv8 backbone作为特征提取器
    - 冻结参数（或在初期冻结）
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        
        # 这里简化实现，实际应该加载预训练的YOLOv8模型
        # 为了演示，我们创建一个简化的backbone
        
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Stage 1
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        
        # Detection head (simplified)
        self.detect_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, num_classes + 4, 1)  # classes + bbox
        )
        
        # 是否冻结参数
        if pretrained:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: 干净图像 [B, C, H, W]
        Returns:
            features: 语义特征字典
            detections: 检测结果
        """
        features = {}
        
        # Backbone
        feat = self.backbone(x)
        features['semantic'] = feat
        
        # Detection head
        detections = self.detect_head(feat)
        features['detections'] = detections
        
        return detections, features


class AdaptiveFeatureBridging(nn.Module):
    """
    自适应特征桥接模块（AFB）
    
    创新点（位置一）：
    在AFB中加入显式的对比学习对齐，而不仅仅是隐式的掩码一致性约束
    
    功能：
    1. 融合IRT教师的重建特征和SPT教师的语义特征
    2. 通过对比学习拉近学生特征与教师特征
    3. 生成对齐后的特征供学生网络使用
    """
    
    def __init__(self, feature_dim=256, use_contrastive=True):
        super().__init__()
        self.use_contrastive = use_contrastive
        
        # 特征投影层（将不同教师的特征映射到统一空间）
        self.irt_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.spt_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.student_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 对比学习投影头（位置一的创新点）
        if use_contrastive:
            self.contrastive_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim // 2)
            )
    
    def forward(self, irt_feat, spt_feat, student_feat, targets=None):
        """
        Args:
            irt_feat: IRT教师的特征 [B, C, H, W]
            spt_feat: SPT教师的特征 [B, C, H, W]
            student_feat: 学生网络的特征 [B, C, H', W']
            targets: 目标标签（用于对比学习的正负样本构建）
        Returns:
            aligned_feat: 对齐后的特征 [B, C, H, W]
            contrastive_loss: 对比学习损失（如果use_contrastive=True）
        """
        # 空间对齐：将student_feat上采样到与irt_feat相同的尺寸
        if student_feat.shape[2:] != irt_feat.shape[2:]:
            student_feat = F.interpolate(
                student_feat, 
                size=irt_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 同样处理spt_feat
        if spt_feat.shape[2:] != irt_feat.shape[2:]:
            spt_feat = F.interpolate(
                spt_feat, 
                size=irt_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 特征投影
        irt_proj = self.irt_proj(irt_feat)
        spt_proj = self.spt_proj(spt_feat)
        student_proj = self.student_proj(student_feat)
        
        # 特征融合
        fused = torch.cat([irt_proj, spt_proj, student_proj], dim=1)
        aligned_feat = self.fusion(fused)
        
        # 对比学习对齐（位置一的创新点）
        contrastive_loss = torch.tensor(0.0, device=aligned_feat.device)
        if self.use_contrastive and self.training:
            # 将空间特征展平
            B, C, H, W = aligned_feat.shape
            
            # 全局平均池化得到全局特征
            irt_global = F.adaptive_avg_pool2d(irt_proj, 1).view(B, C)
            spt_global = F.adaptive_avg_pool2d(spt_proj, 1).view(B, C)
            student_global = F.adaptive_avg_pool2d(student_proj, 1).view(B, C)
            
            # 对比学习投影
            irt_emb = self.contrastive_head(irt_global)
            spt_emb = self.contrastive_head(spt_global)
            student_emb = self.contrastive_head(student_global)
            
            # 计算对比学习损失
            # 学生特征应该靠近两个教师特征的均值
            teacher_mean = (irt_emb + spt_emb) / 2
            
            # InfoNCE风格的对比损失
            pos_sim = F.cosine_similarity(student_emb, teacher_mean, dim=-1)
            neg_sim_irt = F.cosine_similarity(student_emb, irt_emb.detach(), dim=-1)
            neg_sim_spt = F.cosine_similarity(student_emb, spt_emb.detach(), dim=-1)
            
            temperature = 0.07
            pos_exp = torch.exp(pos_sim / temperature)
            neg_exp = torch.exp(neg_sim_irt / temperature) + torch.exp(neg_sim_spt / temperature)
            
            contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
        
        return aligned_feat, contrastive_loss


class DTFAStudentNetwork(nn.Module):
    """
    学生网络（基于YOLOv8-FCA增强）
    
    功能：
    1. 接收对齐后的特征进行检测
    2. 通过三重损失进行训练：
       - 检测损失（detection loss）
       - 重建一致性损失（reconstruction consistency）
       - 特征级对比损失（feature-level contrastive loss，位置二的创新点）
    """
    
    def __init__(self, num_classes=4, use_fca_cl=True):
        super().__init__()
        self.use_fca_cl = use_fca_cl
        
        # Backbone (简化版，实际应使用完整的YOLOv8 backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
        # AFB模块
        self.afb = AdaptiveFeatureBridging(feature_dim=256, use_contrastive=True)
        
        # Detection head
        self.detect_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, num_classes + 4, 1)
        )
    
    def forward(self, x, irt_feat, spt_feat, targets=None):
        """
        Args:
            x: 输入图像（退化图像）
            irt_feat: IRT教师特征
            spt_feat: SPT教师特征
            targets: 目标标签
        Returns:
            detections: 检测结果
            afb_contrastive_loss: AFB中的对比学习损失
            student_feat: 学生特征（用于特征级对比损失）
        """
        # Backbone
        student_feat = self.backbone(x)
        
        # AFB特征对齐（位置一）
        aligned_feat, afb_contrastive_loss = self.afb(
            irt_feat, spt_feat, student_feat, targets
        )
        
        # Detection head
        detections = self.detect_head(aligned_feat)
        
        return detections, afb_contrastive_loss, student_feat


class DTFALoss(nn.Module):
    """
    DTFA总损失函数
    
    包含三个部分：
    1. 检测损失（Detection Loss）
    2. 重建损失（Reconstruction Loss）
    3. 特征级对比损失（Feature-level Contrastive Loss，位置二的创新点）
    """
    
    def __init__(self, lambda_recon=1.0, lambda_contrastive=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_contrastive = lambda_contrastive
        
        # 重建损失（L1 + SSIM）
        self.l1_loss = nn.L1Loss()
        
        # 检测损失（这里简化为MSE，实际应使用YOLO的损失）
        self.det_loss = nn.MSELoss()
    
    def forward(self, recon_img, recon_target, det_pred, det_target, 
                student_feat, irt_feat, spt_feat, afb_contrastive_loss=None):
        """
        Args:
            recon_img: 重建图像
            recon_target: 目标干净图像
            det_pred: 检测预测
            det_target: 检测目标
            student_feat: 学生特征
            irt_feat: IRT教师特征
            spt_feat: SPT教师特征
            afb_contrastive_loss: AFB中的对比学习损失
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 1. 重建损失
        recon_loss = self.l1_loss(recon_img, recon_target)
        
        # 2. 检测损失
        det_loss = self.det_loss(det_pred, det_target)
        
        # 3. 特征级对比损失（位置二的创新点）
        # 学生特征应该同时靠近IRT和SPT教师特征
        # 首先确保所有特征有相同的空间尺寸
        target_spatial_size = irt_feat.shape[2:]
        
        # 对齐student_feat到irt_feat的尺寸
        if student_feat.shape[2:] != target_spatial_size:
            student_feat_aligned = F.interpolate(
                student_feat, size=target_spatial_size, mode='bilinear', align_corners=False
            )
        else:
            student_feat_aligned = student_feat
        
        # 归一化特征
        student_feat_norm = F.normalize(student_feat_aligned.flatten(1), dim=-1)
        # 只detach教师特征，保持学生特征的梯度
        irt_feat_norm = F.normalize(irt_feat.detach().flatten(1), dim=-1)
        spt_feat_norm = F.normalize(spt_feat.detach().flatten(1), dim=-1)
        
        # 计算余弦相似度
        sim_irt = F.cosine_similarity(student_feat_norm, irt_feat_norm, dim=-1).mean()
        sim_spt = F.cosine_similarity(student_feat_norm, spt_feat_norm, dim=-1).mean()
        
        # 对比损失：最大化相似度（最小化距离）
        feature_contrastive_loss = 2.0 - sim_irt - sim_spt
        
        # 4. AFB对比损失（如果存在）
        if afb_contrastive_loss is not None:
            afb_loss = afb_contrastive_loss
        else:
            afb_loss = torch.tensor(0.0, device=recon_img.device)
        
        # 总损失
        total_loss = (
            det_loss + 
            self.lambda_recon * recon_loss + 
            self.lambda_contrastive * feature_contrastive_loss +
            afb_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'detection': det_loss,
            'reconstruction': recon_loss,
            'feature_contrastive': feature_contrastive_loss,
            'afb_contrastive': afb_loss
        }
        
        return total_loss, loss_dict


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("DTFA双教师框架测试")
    print("=" * 80)
    
    # 创建模型
    irt_teacher = IRTTeacher(in_channels=3, base_channels=64, use_fca=True)
    spt_teacher = SPTTeacher(num_classes=4, pretrained=True)
    student = DTFAStudentNetwork(num_classes=4, use_fca_cl=True)
    criterion = DTFALoss(lambda_recon=1.0, lambda_contrastive=0.1)
    
    # 模拟输入
    batch_size = 2
    degraded_img = torch.randn(batch_size, 3, 256, 256)  # 退化图像
    clean_img = torch.randn(batch_size, 3, 256, 256)      # 干净图像（重建目标）
    det_target = torch.randn(batch_size, 8, 256, 256)     # 检测目标
    
    print("\n1. IRT教师前向传播...")
    recon_img, irt_features = irt_teacher(degraded_img)
    print(f"   输入形状: {degraded_img.shape}")
    print(f"   重建图像形状: {recon_img.shape}")
    print(f"   特征键: {list(irt_features.keys())}")
    
    print("\n2. SPT教师前向传播...")
    spt_det, spt_features = spt_teacher(clean_img)
    print(f"   输入形状: {clean_img.shape}")
    print(f"   检测输出形状: {spt_det.shape}")
    print(f"   特征键: {list(spt_features.keys())}")
    
    print("\n3. 学生网络前向传播...")
    # 使用IRT和SPT的特征
    irt_feat = irt_features['enc3']  # 使用encoder第3层特征
    spt_feat = spt_features['semantic']
    
    det_pred, afb_loss, student_feat = student(
        degraded_img, irt_feat, spt_feat, targets=None
    )
    print(f"   检测预测形状: {det_pred.shape}")
    print(f"   学生特征形状: {student_feat.shape}")
    print(f"   AFB对比损失: {afb_loss}")
    
    print("\n4. 计算总损失...")
    total_loss, loss_dict = criterion(
        recon_img, clean_img,
        det_pred, det_target,
        student_feat, irt_feat, spt_feat,
        afb_contrastive_loss=afb_loss
    )
    print(f"   总损失: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value.item():.4f}")
    
    print("\n" + "=" * 80)
    print("测试完成！DTFA框架运行正常")
    print("=" * 80)
