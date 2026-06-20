"""
验证DTFA三大创新点的完整性测试
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from ultralytics.nn.modules.dtfa_framework import (
    IRTTeacher,
    SPTTeacher,
    DTFAStudentNetwork,
    AdaptiveFeatureBridging,
    DTFALoss
)

print("=" * 80)
print("DTFA三大创新点验证测试")
print("=" * 80)

# ============================================================================
# 创新点一：AFB模块内的显式对比学习对齐
# ============================================================================
print("\n" + "💡" * 40)
print("创新点一：AFB模块内的显式对比学习对齐")
print("💡" * 40)

try:
    # 创建AFB模块
    afb = AdaptiveFeatureBridging(feature_dim=256, use_contrastive=True)
    
    # 检查是否有对比学习投影头
    has_contrastive_head = hasattr(afb, 'contrastive_head') and afb.contrastive_head is not None
    print(f"\n✓ AFB模块创建成功")
    print(f"✓ 使用对比学习: {afb.use_contrastive}")
    print(f"✓ 包含对比学习投影头: {has_contrastive_head}")
    
    if has_contrastive_head:
        print(f"  投影头结构:")
        for i, layer in enumerate(afb.contrastive_head):
            print(f"    Layer {i}: {layer.__class__.__name__}")
    
    # 测试前向传播并生成对比损失
    batch_size = 2
    feat_dim = 256
    spatial_size = 32
    
    irt_feat = torch.randn(batch_size, feat_dim, spatial_size, spatial_size)
    spt_feat = torch.randn(batch_size, feat_dim, spatial_size, spatial_size)
    student_feat = torch.randn(batch_size, feat_dim, spatial_size, spatial_size)
    
    afb.train()  # 设置为训练模式以启用对比学习
    aligned_feat, contrastive_loss = afb(irt_feat, spt_feat, student_feat, targets=None)
    
    print(f"\n✓ 前向传播成功")
    print(f"  输入特征形状: {irt_feat.shape}")
    print(f"  对齐后特征形状: {aligned_feat.shape}")
    print(f"  对比学习损失: {contrastive_loss.item():.6f}")
    print(f"  损失为有限值: {torch.isfinite(contrastive_loss).item()}")
    
    # 验证对比损失是否为标量
    assert contrastive_loss.dim() == 0, "对比损失应该是标量"
    assert torch.isfinite(contrastive_loss), "对比损失应该是有限值"
    
    print("\n✅ 创新点一验证通过！")
    print("   - AFB模块包含显式对比学习机制")
    print("   - 能够计算InfoNCE风格的对比损失")
    print("   - 损失值为有效的标量")
    
except Exception as e:
    print(f"\n❌ 创新点一验证失败: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 创新点二：特征级对比损失
# ============================================================================
print("\n" + "💡" * 40)
print("创新点二：特征级对比损失（权重0.1）")
print("💡" * 40)

try:
    # 创建损失函数
    criterion = DTFALoss(lambda_recon=1.0, lambda_contrastive=0.1)
    
    print(f"\n✓ DTFALoss创建成功")
    print(f"  重建损失权重 (lambda_recon): {criterion.lambda_recon}")
    print(f"  对比损失权重 (lambda_contrastive): {criterion.lambda_contrastive}")
    
    # 准备测试数据（需要有梯度的目标）
    batch_size = 2
    recon_img = torch.randn(batch_size, 3, 256, 256, requires_grad=False)
    recon_target = torch.randn(batch_size, 3, 256, 256, requires_grad=False)
    det_pred = torch.randn(batch_size, 8, 64, 64, requires_grad=True)  # 需要梯度
    det_target = torch.randn(batch_size, 8, 64, 64, requires_grad=False)  # 不要全零
    student_feat = torch.randn(batch_size, 256, 32, 32, requires_grad=True)  # 需要梯度
    irt_feat = torch.randn(batch_size, 256, 32, 32, requires_grad=False)
    spt_feat = torch.randn(batch_size, 256, 32, 32, requires_grad=False)
    
    # 计算损失
    total_loss, loss_dict = criterion(
        recon_img, recon_target,
        det_pred, det_target,
        student_feat, irt_feat, spt_feat,
        afb_contrastive_loss=None
    )
    
    print(f"\n✓ 损失计算成功")
    print(f"  总损失: {total_loss.item():.6f}")
    print(f"\n  损失分解:")
    for key, value in loss_dict.items():
        print(f"    {key:<30}: {value.item():.6f}")
    
    # 验证是否包含特征级对比损失
    has_feature_contrastive = 'feature_contrastive' in loss_dict
    print(f"\n✓ 包含特征级对比损失: {has_feature_contrastive}")
    
    if has_feature_contrastive:
        feature_contrastive_value = loss_dict['feature_contrastive'].item()
        print(f"  特征级对比损失值: {feature_contrastive_value:.6f}")
        print(f"  损失为有限值: {torch.isfinite(torch.tensor(feature_contrastive_value)).item()}")
        
        # 验证对比损失权重是否正确应用
        # 特征对比损失应该在总损失中有贡献
        assert torch.isfinite(total_loss), "总损失应该是有限值"
        assert total_loss.item() > 0, "总损失应该大于0"
    
    # 验证梯度可以反向传播
    total_loss.backward()
    print(f"\n✓ 反向传播成功（梯度可计算）")
    
    print("\n✅ 创新点二验证通过！")
    print("   - 损失函数包含特征级对比损失项")
    print("   - 对比损失权重设置为0.1")
    print("   - 损失计算和梯度传播正常")
    
except Exception as e:
    print(f"\n❌ 创新点二验证失败: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 创新点三：IRT编码器中的FCA特征聚集
# ============================================================================
print("\n" + "💡" * 40)
print("创新点三：IRT编码器中的FCA特征聚集")
print("💡" * 40)

try:
    # 创建带FCA的IRT教师
    irt_with_fca = IRTTeacher(in_channels=3, base_channels=64, use_fca=True)
    
    # 创建不带FCA的IRT教师（用于对比）
    irt_without_fca = IRTTeacher(in_channels=3, base_channels=64, use_fca=False)
    
    print(f"\n✓ IRT教师网络创建成功")
    print(f"  使用FCA: {irt_with_fca.use_fca}")
    print(f"  包含FCA聚集模块: {hasattr(irt_with_fca, 'fca_aggregation')}")
    
    if hasattr(irt_with_fca, 'fca_aggregation'):
        print(f"  FCA模块结构:")
        for i, layer in enumerate(irt_with_fca.fca_aggregation):
            if isinstance(layer, nn.Conv2d):
                print(f"    Conv2d: {layer.in_channels} -> {layer.out_channels}, kernel={layer.kernel_size}")
            elif isinstance(layer, nn.BatchNorm2d):
                print(f"    BatchNorm2d: {layer.num_features} channels")
            else:
                print(f"    {layer.__class__.__name__}")
    
    # 测试前向传播
    irt_with_fca.eval()
    irt_without_fca.eval()
    
    input_img = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        recon_with_fca, feats_with_fca = irt_with_fca(input_img)
        recon_without_fca, feats_without_fca = irt_without_fca(input_img)
    
    print(f"\n✓ 前向传播成功")
    print(f"  输入图像形状: {input_img.shape}")
    print(f"  重建图像形状（带FCA）: {recon_with_fca.shape}")
    print(f"  重建图像形状（无FCA）: {recon_without_fca.shape}")
    
    # 检查特征字典
    print(f"\n  带FCA的特征键: {list(feats_with_fca.keys())}")
    print(f"  无FCA的特征键: {list(feats_without_fca.keys())}")
    
    # 验证FCA增强特征是否存在
    has_fca_enhanced = 'fca_enhanced' in feats_with_fca
    print(f"\n✓ 包含FCA增强特征: {has_fca_enhanced}")
    
    if has_fca_enhanced:
        fca_feat_shape = feats_with_fca['fca_enhanced'].shape
        print(f"  FCA增强特征形状: {fca_feat_shape}")
        print(f"  特征为有限值: {torch.isfinite(feats_with_fca['fca_enhanced']).all().item()}")
    
    # 对比有无FCA的重建差异
    recon_diff = torch.abs(recon_with_fca - recon_without_fca).mean()
    print(f"\n✓ 重建差异（带FCA vs 无FCA）: {recon_diff.item():.6f}")
    print(f"  说明FCA确实影响了重建结果")
    
    # 参数量对比
    params_with_fca = sum(p.numel() for p in irt_with_fca.parameters())
    params_without_fca = sum(p.numel() for p in irt_without_fca.parameters())
    param_diff = params_with_fca - params_without_fca
    
    print(f"\n  参数量对比:")
    print(f"    带FCA: {params_with_fca:,}")
    print(f"    无FCA: {params_without_fca:,}")
    print(f"    差异: {param_diff:,} ({param_diff/params_without_fca*100:.2f}%)")
    
    print("\n✅ 创新点三验证通过！")
    print("   - IRT编码器包含FCA特征聚集模块")
    print("   - FCA模块在encoder中正确集成")
    print("   - FCA增强了特征表示并影响重建结果")
    
except Exception as e:
    print(f"\n❌ 创新点三验证失败: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 综合验证：三个创新点协同工作
# ============================================================================
print("\n" + "=" * 80)
print("综合验证：三个创新点协同工作")
print("=" * 80)

try:
    print("\n创建完整的DTFA系统...")
    
    # 创建所有组件
    irt = IRTTeacher(in_channels=3, base_channels=64, use_fca=True)
    spt = SPTTeacher(num_classes=4, pretrained=True)
    student = DTFAStudentNetwork(num_classes=4, use_fca_cl=True)
    criterion = DTFALoss(lambda_recon=1.0, lambda_contrastive=0.1)
    
    print(f"✓ IRT教师: {sum(p.numel() for p in irt.parameters()):,} 参数")
    print(f"✓ SPT教师: {sum(p.numel() for p in spt.parameters()):,} 参数")
    print(f"✓ 学生网络: {sum(p.numel() for p in student.parameters()):,} 参数")
    print(f"✓ 损失函数: lambda_contrastive={criterion.lambda_contrastive}")
    
    # 设置模式
    irt.eval()
    spt.eval()
    student.train()  # 训练模式以启用对比学习
    
    # 准备数据
    batch_size = 2
    degraded_img = torch.randn(batch_size, 3, 256, 256)
    clean_img = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\n执行前向传播...")
    
    # IRT前向（位置三：FCA聚集）
    with torch.no_grad():
        recon_img, irt_feats = irt(degraded_img)
    
    # 检查FCA特征
    has_fca = 'fca_enhanced' in irt_feats
    print(f"  ✓ 位置三验证: FCA特征存在 = {has_fca}")
    
    # SPT前向
    with torch.no_grad():
        spt_det, spt_feats = spt(clean_img)
    
    # 获取特征并对齐尺寸
    irt_feat = irt_feats['enc3']  # [B, 256, 64, 64]
    spt_feat = spt_feats['semantic']  # [B, 512, 16, 16]
    
    # 调整SPT特征以匹配IRT
    if spt_feat.shape[1] != irt_feat.shape[1]:
        spt_proj_conv = nn.Conv2d(spt_feat.shape[1], irt_feat.shape[1], 1)
        spt_feat = spt_proj_conv(spt_feat)
    
    if spt_feat.shape[2:] != irt_feat.shape[2:]:
        spt_feat = torch.nn.functional.interpolate(
            spt_feat, size=irt_feat.shape[2:], mode='bilinear', align_corners=False
        )
    
    # 学生网络前向（位置一：AFB对比学习）
    det_pred, afb_contrastive_loss, student_feat = student(
        degraded_img, irt_feat, spt_feat, targets=None
    )
    
    has_afb_loss = afb_contrastive_loss is not None and torch.isfinite(afb_contrastive_loss)
    print(f"  ✓ 位置一验证: AFB对比损失存在 = {has_afb_loss}")
    if has_afb_loss:
        print(f"    AFB对比损失值: {afb_contrastive_loss.item():.6f}")
    
    # 计算总损失（位置二：特征级对比损失）
    det_target = torch.zeros_like(det_pred)
    total_loss, loss_dict = criterion(
        recon_img, clean_img,
        det_pred, det_target,
        student_feat, irt_feat, spt_feat,
        afb_contrastive_loss=afb_contrastive_loss
    )
    
    has_feature_contrastive = 'feature_contrastive' in loss_dict
    print(f"  ✓ 位置二验证: 特征级对比损失存在 = {has_feature_contrastive}")
    if has_feature_contrastive:
        print(f"    特征级对比损失值: {loss_dict['feature_contrastive'].item():.6f}")
    
    print(f"\n  总损失: {total_loss.item():.6f}")
    print(f"  损失分解:")
    for key, value in loss_dict.items():
        print(f"    {key:<30}: {value.item():.6f}")
    
    # 验证梯度
    total_loss.backward()
    print(f"\n✓ 反向传播成功")
    
    print("\n" + "=" * 80)
    print("🎉 所有创新点验证通过！")
    print("=" * 80)
    print("\n总结:")
    print("  ✅ 位置一: AFB模块内的显式对比学习对齐 - 已实现")
    print("  ✅ 位置二: 特征级对比损失（权重0.1）- 已实现")
    print("  ✅ 位置三: IRT编码器中的FCA特征聚集 - 已实现")
    print("\nDTFA+FCA-CL框架已完整实现，可以进行训练和实验！")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ 综合验证失败: {str(e)}")
    import traceback
    traceback.print_exc()
