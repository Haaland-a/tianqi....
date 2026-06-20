# 双教师知识蒸馏恶劣天气目标检测方案

## 整体架构

```
                     ┌──────────────────────┐
                     │  SPT 语义先验教师      │
                     │  (预训练雾天 YOLO-FCA) │ ← 全程冻结, 无梯度
                     │  输出: Neck 语义特征    │
                     └──────────┬───────────┘
                                │ SPT_Distill_Loss (MSE)
                                ▼
  恶劣天气原图 ──┬─────→  YOLO 学生网络 (FCA) ──→ 检测结果 (Det_Loss)
                │          唯一可训练网络
                │              ▲
                │    IRT_Distill_Loss (MSE)
                ▼              │
          ┌─────────────┐      │
          │ IRT 重建教师  │      │
          │ (AOD-Net)    │ ──→ 输出: 复原图 + 边缘特征
          │ 全程冻结      │
          └─────────────┘
```

## 三大网络角色

### SPT 语义先验教师
- **权重**: 预训练雾天 YOLO-FCA 模型 (`best.pt`)
- **输出**: Neck 层 P3/P4/P5 语义特征
- **状态**: 全程冻结, `requires_grad=False`

### IRT 不变重建教师
- **模型**: AOD-Net 轻量去雾网络 (1.7K 参数)
- **输入**: 任意恶劣天气图像
- **输出**: 复原清晰图像 + Sobel 边缘轮廓特征
- **状态**: 全程冻结
- **预训练**: 需下载 AOD-Net 公开预训练权重

### YOLO 学生网络
- **基底**: YOLOv8 + FCA (跨层特征聚集)
- **输入**: 恶劣天气原图
- **输出**: 检测结果 + Neck 特征
- **状态**: 唯一可训练网络
- **训练**: 两阶段

## 前向传播逻辑

```
Step 1: 天气原图 I → IRT(AOD-Net) → 复原图 I' + Sobel边缘特征 E_irt
Step 2: 天气原图 I → SPT(YOLO-FCA) → Neck语义特征 [F_P3, F_P4, F_P5]
Step 3: 天气原图 I → 学生(YOLO-FCA) → 检测输出 + Neck特征 [S_P3, S_P4, S_P5]
Step 4: 计算损失
        Det_Loss = YOLO原生检测损失
        IRT_Distill = MSE(S_P3, E_irt)  ← 学生低层特征对齐边缘
        SPT_Distill = MSE([S_P3,S_P4,S_P5], [F_P3,F_P4,F_P5])  ← 多层语义对齐
        Total = Det + 0.3*IRT + 1.0*SPT
Step 5: 反向传播仅更新学生参数
```

## 损失函数权重配比

| 超参 | 默认值 | 说明 | 调优方向 |
|------|--------|------|---------|
| λ1 (IRT) | 0.3 | IRT 蒸馏权重 | 检测不稳定/边缘模糊 → 增大到 0.5-1.0 |
| λ2 (SPT) | 1.0 | SPT 蒸馏权重 | 检测精度不足 → 增大到 1.5-2.0 |

## 两阶段训练策略

### Stage 1: Neck/Head Adaptation (30 epochs, LR=1e-3)
- **冻结**: Student Backbone (layers 0-9) + SPT + IRT
- **训练**: Student Neck + FCA + Detect Head
- **目标**: Neck 学习融合双教师特征, FCA 自适应

### Stage 2: Global Fine-Tuning (50 epochs, LR=1e-4)
- **冻结**: 仅 SPT + IRT
- **训练**: Student 全部参数
- **目标**: 全局微调, 消除两阶段之间的特征不匹配

## 文件说明

```
distillation/
├── __init__.py              # 模块入口
├── aod_net.py               # IRT 教师: AOD-Net + Sobel 边缘提取
├── spt_teacher.py           # SPT 教师: YOLO-FCA 语义特征提取
├── losses.py                # 组合蒸馏损失 (Det + IRT + SPT)
├── dataset.py               # 成对数据加载器
├── trainer.py               # 双教师蒸馏训练器 (两阶段)
├── train_dual_teacher.py    # 主训练入口 + 全部超参数
├── setup_dataset.py         # 数据集 8:1:1 拆分脚本
├── data_weather.yaml        # YOLO 数据配置
└── README.md                # 本文件
```

## 运行步骤

### 1. 环境依赖
```bash
pip install torch>=2.0 ultralytics>=8.0 opencv-python numpy tqdm
```

### 2. 准备数据集
```bash
cd E:\Personal\Desktop\learn (2)\learn\learn\distillation
python setup_dataset.py
```
生成 `adverse_weather_yolo/` (约 610 张图片, 8:1:1 拆分)

### 3. 训练基础 YOLO-FCA 模型 (获取 SPT 教师权重)
```bash
cd ..
python train.py
```
确保 `runs/fca_detect/train_fca_cl/weights/best.pt` 存在

### 4. 下载 AOD-Net 预训练权重 (可选)
从 https://github.com/weichen582/AOD-Net 下载预训练模型

### 5. 运行双教师蒸馏训练
```bash
cd distillation
python train_dual_teacher.py
```

## 训练超参数参考

| 参数 | Stage 1 | Stage 2 | 说明 |
|------|---------|---------|------|
| epochs | 30 | 50 | 训练轮数 |
| lr | 1e-3 | 1e-4 | 学习率 |
| batch | 8 | 8 | 显存 6GB 可用 |
| imgsz | 640 | 640 | 输入尺寸 |
| optimizer | AdamW | AdamW | 优化器 |
| amp | Yes | Yes | 混合精度 |
| λ_IRT | 0.3 | 0.3 | IRT 蒸馏权重 |
| λ_SPT | 1.0 | 1.0 | SPT 蒸馏权重 |

## 常见问题排查

### KeyError: 'FCA'
`sys.path` 未正确指向 `ultralytics-main`。确保代码首行执行:
```python
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'ultralytics-main'))
```

### No labels found
图片和标签目录名必须为 `images` 和 `labels`。YOLO 硬编码替换路径中的 `images` → `labels`。

### OOM (Out of Memory)
- 降低 `batch_size` 到 4
- 降低 `img_size` 到 416
- 关闭 `amp`

### SPT 教师权重不存在的降级
如果 SPT 权重路径无效,训练器会警告但仍可运行。IRT 蒸馏仍生效,但 SPT 蒸馏损失为 0。

### 双教师前向不同步
确保 IRT/SPS 输入完全相同的预处理图像。使用同一 DataLoader 输出。
