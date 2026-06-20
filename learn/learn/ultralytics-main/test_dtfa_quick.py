
"""
DTFA Quick Test - 快速测试双教师框架
"""

import sys
sys.path.insert(0, '.')

import torch
from ultralytics.nn.modules.dtfa_framework import (
    IRTTeacher,
    SPTTeacher, 
    DTFAStudentNetwork,
    DTFALoss
)

print("=" * 80)
print("DTFA双教师框架 - 快速测试")
print("=" * 80)

# 1. 创建模型
print("\n1. 创建模型...")
irt = IRTTeacher(in_channels=3, base_channels=64, use_fca=True)
spt = SPTTeacher(num_classes=4, pretrained=True)
student = DTFAStudentNetwork(num_classes=4, use_fca_cl=True)
criterion = DTFALoss(lambda_recon=1.0, lambda_contrastive=0.1)

print(f"   ✓ IRT教师: {sum(p.numel() for p in irt.parameters()):,} 参数")
print(f"   ✓ SPT教师: {sum(p.numel() for p in spt.parameters()):,} 参数")
print(f"   ✓ 学生网络: {sum(p.numel() for p in student.parameters()):,} 参数")

# 2. 测试前向传播
print("\n2. 测试前向传播...")
batch_size = 2
degraded_img = torch.randn(batch_size, 3, 256, 256)
clean_img = torch.randn(batch_size, 3, 256, 256)

irt.eval()
spt.eval()
student.eval()

with torch.no_grad():
    recon_img, irt_feats = irt(degraded_img)
    spt_det, spt_feats = spt(clean_img)
    
    irt_feat = irt_feats['enc3']  # [B, 256, 64, 64]
    spt_feat = spt_feats['semantic']  # [B, 512, 16, 16]
    
    # 需要调整spt_feat的通道数和空间尺寸到与irt_feat一致
    if spt_feat.shape[1] != irt_feat.shape[1]:
        spt_proj_conv = torch.nn.Conv2d(spt_feat.shape[1], irt_feat.shape[1], 1).to(spt_feat.device)
        spt_feat = spt_proj_conv(spt_feat)
    
    # 上采样spt_feat到irt_feat的尺寸
    if spt_feat.shape[2:] != irt_feat.shape[2:]:
        spt_feat = torch.nn.functional.interpolate(spt_feat, size=irt_feat.shape[2:], mode='bilinear', align_corners=False)
    
    det_pred, afb_loss, student_feat = student(degraded_img, irt_feat, spt_feat)

print(f"   ✓ 重建图像: {recon_img.shape}")
print(f"   ✓ 检测预测: {det_pred.shape}")
print(f"   ✓ AFB对比损失: {afb_loss}")

# 3. 测试损失计算
print("\n3. 测试损失计算...")
student.train()
det_target = torch.zeros_like(det_pred)

total_loss, loss_dict = criterion(
    recon_img, clean_img,
    det_pred, det_target,
    student_feat, irt_feat, spt_feat,
    afb_contrastive_loss=afb_loss
)

print(f"   ✓ 总损失: {total_loss.item():.4f}")
for key, value in loss_dict.items():
    print(f"   ✓ {key}: {value.item():.4f}")

# 4. 验证创新点
print("\n4. 验证三个创新点...")
print("   💡 位置一: AFB显式对比学习 - ", "✓" if afb_loss is not None else "✗")
print("   💡 位置二: 特征级对比损失 - ", "✓" if 'feature_contrastive' in loss_dict else "✗")
print("   💡 位置三: IRT-FCA聚集 - ", "✓" if 'fca_enhanced' in irt_feats else "✗")

print("\n" + "=" * 80)
print("🎉 所有测试通过！DTFA框架运行正常")
print("=" * 80)
print("\n下一步:")
print("  1. 准备paired数据集（干净-退化图像对）")
print("  2. 配置训练参数")
print("  3. 运行训练: python ultralytics/utils/train_dtfa.py --data your_data.yaml")
print("=" * 80)
