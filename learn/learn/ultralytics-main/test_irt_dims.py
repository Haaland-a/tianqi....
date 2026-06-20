
import sys
sys.path.insert(0, '.')

import torch
from ultralytics.nn.modules.dtfa_framework import IRTTeacher

print("测试IRT教师网络的维度...")

irt = IRTTeacher(in_channels=3, base_channels=64, use_fca=True)
irt.eval()

degraded_img = torch.randn(1, 3, 256, 256)
print(f"输入: {degraded_img.shape}")

with torch.no_grad():
    # Encoder
    enc1 = irt.enc_conv1(degraded_img)
    print(f"enc1: {enc1.shape}")
    
    enc2 = irt.enc_conv2(enc1)
    print(f"enc2: {enc2.shape}")
    
    enc3 = irt.enc_conv3(enc2)
    print(f"enc3: {enc3.shape}")
    
    enc4 = irt.enc_conv4(enc3)
    print(f"enc4: {enc4.shape}")
    
    # Bottleneck
    bottleneck = irt.bottleneck(enc4)
    print(f"bottleneck: {bottleneck.shape}")
    
    # Decoder
    dec1 = irt.dec_upsample1(bottleneck)
    print(f"dec1 (after upsample): {dec1.shape}")
    
    enc4_up = torch.nn.functional.interpolate(enc4, size=dec1.shape[2:])
    print(f"enc4_up: {enc4_up.shape}")
    
    dec1_cat = torch.cat([dec1, enc4_up], dim=1)
    print(f"dec1_cat: {dec1_cat.shape}")
    
    dec1_out = irt.dec_conv1(dec1_cat)
    print(f"dec1_out: {dec1_out.shape}")
    
    dec2 = irt.dec_upsample2(dec1_out)
    print(f"dec2 (after upsample): {dec2.shape}")
    
    # FCA
    enc3_up = torch.nn.functional.interpolate(enc3, size=enc2.shape[2:])
    fused = torch.cat([enc2, enc3_up], dim=1)
    fca_feat = irt.fca_aggregation(fused)
    print(f"fca_feat: {fca_feat.shape}")
    
    dec2_cat = torch.cat([dec2, fca_feat], dim=1)
    print(f"dec2_cat: {dec2_cat.shape}")
    
    dec2_out = irt.dec_conv2(dec2_cat)
    print(f"dec2_out: {dec2_out.shape}")
    
    dec3 = irt.dec_upsample3(dec2_out)
    print(f"dec3 (after upsample): {dec3.shape}")
    print(f"enc2: {enc2.shape}")
    
    print(f"\n❌ 维度不匹配！dec3={dec3.shape}, enc2={enc2.shape}")
