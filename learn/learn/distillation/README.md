# 双教师知识蒸馏恶劣天气目标检测方案

## 一、整体方案说明

本方案采用**双教师知识蒸馏**架构，在恶劣天气（雾、霾、雨、雪）场景下提升 YOLO 目标检测精度。核心思路：用两个预训练的教师网络指导一个学生 YOLO 网络学习，使学生在低质量天气图像上也能提取鲁棒特征。

### 整体架构图

```
                     ┌─────────────────────────────┐
                     │  SPT 语义先验教师             │
                     │  (预训练雾天 YOLO-FCA 模型)    │ ← 全程冻结，关闭梯度
                     │  输出: Neck 层 P3/P4/P5 语义特征 │
                     └──────────────┬──────────────┘
                                    │ SPT 蒸馏损失 (MSE)
                                    ▼
      恶劣天气原图 ──┬───────→ YOLO 学生网络 (含 FCA 模块) ──→ 检测结果 (原生检测损失)
                     │              唯一可训练网络
                     │                    ▲
                     │      IRT 蒸馏损失 (MSE)
                     ▼                    │
               ┌─────────────┐           │
               │ IRT 不变重建教师│           │
               │ (AOD-Net 去雾) │ ───────→ 输出: 复原清晰图 + Sobel 边缘特征
               │ 全程冻结       │
               └─────────────┘
```

### 三大网络角色与规则

**SPT 语义先验教师**（语义引导）
- 加载单独训练完成的雾天 YOLO-FCA 模型权重（`best.pt`）
- 前向传播时捕获 Neck 层中间特征（P3/P4/P5 三层），用于特征蒸馏
- 全程冻结所有参数，`requires_grad=False`，仅做前向推理
- 作用：向学生提供"在雾天场景下什么特征是有判别力的"语义先验

**IRT 不变重建教师**（结构约束）
- 使用 AOD-Net 轻量去雾网络，直接加载公开预训练权重
- 输入任意恶劣天气原图（雾/霾/雨/雪），输出复原后的清晰图像
- 通过 Sobel 算子从复原图中提取底层轮廓/边缘特征
- 全程冻结参数，关闭梯度，仅做前向推理
- 作用：向学生提供"清晰场景下物体应有的底层结构"信息

**YOLO 学生网络**（唯一可训练）
- 基底与 SPT 同版本的 YOLOv8 模型
- Neck 层嵌入 FCA（跨层特征聚集注意力）模块，增强小目标感知
- 分两阶段训练：先冻结主干适应颈部，再全局解冻微调

### 网络前向传播逻辑

```
第1步: 天气原图 → IRT 教师(AOD-Net) → 复原清晰图 + Sobel 边缘特征 E_irt
第2步: 天气原图 → SPT 教师(YOLO-FCA) → Neck 语义特征 [F_P3, F_P4, F_P5]
第3步: 天气原图 → 学生网络(YOLO-FCA) → 检测输出 + Neck 特征 [S_P3, S_P4, S_P5]
第4步: 计算损失
       检测损失 = YOLO 原生分类+回归+置信度损失
       IRT蒸馏损失 = MSE(S_P3, E_irt)           ← 学生低层特征对齐 IRT 边缘
       SPT蒸馏损失 = MSE([S_P3,S_P4,S_P5], [F_P3,F_P4,F_P5])  ← 多层语义对齐
       总损失 = 检测损失 + 0.3×IRT蒸馏 + 1.0×SPT蒸馏
第5步: 反向传播，仅更新学生网络参数
```

## 二、损失函数设计

### 总损失公式

```
Total_Loss = Det_Loss + λ₁ × IRT_Distill_Loss + λ₂ × SPT_Distill_Loss
```

### 各损失说明

| 损失项 | 含义 | 计算方式 |
|--------|------|---------|
| Det_Loss | YOLO 原生检测损失 | 分类损失 + 边界框回归损失 + DFL 置信度损失 |
| IRT_Distill_Loss | 学生特征与 IRT 边缘特征的 MSE | 自适应特征对齐投影后计算均方误差 |
| SPT_Distill_Loss | 学生特征与 SPT 语义特征的 MSE | 多尺度（P3/P4/P5）加权 MSE |

### 权重配比与调优

| 超参 | 默认值 | 含义 | 调优方向 |
|------|--------|------|---------|
| λ₁ (IRT) | **0.3** | IRT 重建蒸馏权重 | 学生检测结果中物体边缘模糊、定位不准 → 增大到 0.5~1.0 |
| λ₂ (SPT) | **1.0** | SPT 语义蒸馏权重 | 学生检测精度 (mAP) 不如预期 → 增大到 1.5~2.0 |
| 对齐通道数 | **128** | 特征投影空间维度 | 训练初期损失震荡 → 降低权重（如 λ₁=0.1, λ₂=0.5）并用 warmup |

**设计原则**：λ₂ > λ₁，因为高层语义特征对检测任务的帮助大于底层边缘特征。

## 三、分阶段训练策略

### 第一阶段：颈部/头部自适应（30 轮，学习率 1×10⁻³）

| 项目 | 状态 |
|------|------|
| **冻结** | 学生 Backbone 主干网络（第 0~9 层）+ SPT 教师 + IRT 教师 |
| **可训练** | 学生 Neck 层 + FCA 特征聚集模块 + Detect 检测头 |
| **优化器** | AdamW，学习率 1×10⁻³，余弦退火调度 |
| **目标** | Neck/FCA 学习融合双教师特征，FCA 自适应天气退化 |

### 第二阶段：全局微调（50 轮，学习率 1×10⁻⁴）

| 项目 | 状态 |
|------|------|
| **冻结** | 仅保留 SPT 教师 + IRT 教师 |
| **可训练** | 学生 YOLO **全部网络参数** |
| **优化器** | AdamW，学习率 1×10⁻⁴（比第一阶段低一个数量级），余弦退火 |
| **目标** | 全局微调解耦，消除两阶段之间的特征不匹配 |

## 四、文件说明

```
distillation/
├── __init__.py              # 模块封装入口
├── aod_net.py               # IRT 教师: AOD-Net 去雾网络 + Sobel 边缘特征提取器
├── spt_teacher.py           # SPT 教师: YOLO-FCA 语义特征提取 (Hook 捕获 Neck 层)
├── losses.py                # 组合蒸馏损失函数 (Det_Loss + IRT_MSE + SPT_MSE)
├── dataset.py               # 成对数据加载器 (天气原图 + 标签)
├── trainer.py               # 双教师蒸馏训练器 (两阶段训练核心逻辑)
├── train_dual_teacher.py    # 主训练入口脚本 + 全部可配置超参数
├── setup_dataset.py         # 数据集预处理脚本 (8:1:1 随机拆分)
├── data_weather.yaml        # YOLO 格式数据集配置文件
└── README.md                # 本文档
```

## 五、运行步骤

### 第1步：安装环境依赖

```bash
pip install torch>=2.0 ultralytics>=8.0 opencv-python numpy tqdm
```

验证：
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 第2步：准备数据集（8:1:1 拆分）

```bash
cd E:\Personal\Desktop\learn (2)\learn\learn\distillation
python setup_dataset.py
```

执行后生成 `E:\Personal\Desktop\learn (2)\adverse_weather_yolo\`，结构如下：
```
adverse_weather_yolo/
├── images/
│   ├── train/    (488 张，80%)
│   ├── val/      (61 张，10%)
│   └── test/     (61 张，10%)
└── labels/
    ├── train/    (488 个 .txt)
    ├── val/      (61 个 .txt)
    └── test/     (61 个 .txt)
```

### 第3步：训练 SPT 教师（基础 YOLO-FCA 模型）

```bash
cd E:\Personal\Desktop\learn (2)\learn\learn
python train.py
```

等待训练完成，确认 `runs/fca_detect/train_fca_cl/weights/best.pt` 存在。


### 第5步：启动双教师蒸馏训练

```bash
cd E:\Personal\Desktop\learn (2)\learn\learn\distillation
python train_dual_teacher.py
```

训练过程输出示例：
```
======================================================================
双教师知识蒸馏 — 恶劣天气目标检测训练
======================================================================
S1 Train [1/30]: 100%|██████████| 97/97 [03:12<00:00]
S1 Epoch   1/30 | LR 1.00e-03 | Loss 4.2135 | Det 2.8934 | IRT 0.0234 | SPT 0.0512
...
S2 Epoch  50/50 | LR 1.00e-05 | Loss 1.9834 | Det 1.8734 | IRT 0.0121 | SPT 0.0323
训练完成! 最终模型: distill_output/stage2_final.pt
```

## 六、训练超参数参考

| 参数 | 第一阶段 | 第二阶段 | 说明 |
|------|---------|---------|------|
| 训练轮数 | 30 | 50 | 小数据集可适当减少 |
| 学习率 | 1×10⁻³ | 1×10⁻⁴ | 第二阶段降一个数量级 |
| 批次大小 | 8 | 8 | 6GB 显存可用，OOM 则降为 4 |
| 输入尺寸 | 640 | 640 | 显存不足可降到 416 |
| 优化器 | AdamW | AdamW | 权重衰减 1×10⁻⁴ |
| 学习率调度 | 余弦退火 | 余弦退火 | 平滑衰减 |
| 混合精度 | 开启 | 开启 | 节省显存，加速训练 |
| λ_IRT | 0.3 | 0.3 | IRT 蒸馏权重 |
| λ_SPT | 1.0 | 1.0 | SPT 蒸馏权重 |
| 数据加载线程 | 4 | 4 | CPU 核心数决定 |
| 早停耐心值 | 10 | 15 | 验证损失不降则提前停止 |

## 七、常见问题排查

### 启动报错 `KeyError: 'FCA'`
`sys.path` 未正确指向修改版 `ultralytics-main`。确认 `train_dual_teacher.py` 首行已执行：
```python
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'ultralytics-main'))
```
注意这行必须在 `from ultralytics import YOLO` 之前。

### 训练日志显示 `No labels found`
图片目录名和标签目录名必须严格为 `images` 和 `labels`。YOLO 内部硬编码将路径中的 `images` 替换为 `labels` 来定位标签文件。如果目录名是 `zuiyou_images` 或 `pictures` 等非标准名称，会导致找不到标签。

### 显存不足 (OOM)
- 将 `batch_size` 从 8 降到 4
- 将 `img_size` 从 640 降到 416
- 关闭混合精度：`amp=False`（虽然会变慢但省显存）

### SPT 教师权重文件不存在
如果 `best.pt` 尚未训练完成，训练器会输出警告但继续运行。此时 IRT 单教师蒸馏仍生效，但 SPT 蒸馏损失为 0。建议先完成基础 YOLO-FCA 模型训练再开始蒸馏。

### 损失值不下降或震荡
- 检查学习率是否过高（第一阶段 1×10⁻³，第二阶段 1×10⁻⁴）
- 降低蒸馏权重：λ₁=0.1, λ₂=0.3
- 增大 `batch_size` 以获得更稳定的梯度
- 检查数据集标签是否正确（类别 ID 是否在 0~3 范围内）

### 训练速度过慢
- 设置 `cache=True`（需要足够内存缓存全部图片到 RAM）
- 增大 `workers`（加载线程数，建议设为 CPU 核心数的一半）
- 使用 `amp=True` 混合精度加速
- 确认 `device='0'` 指向正确的 GPU

### 教师/学生输入不一致
IRT 和 SPT 教师必须接收与训练学生完全相同的预处理后图像。本方案通过同一 DataLoader 输出保证一致性，不需要额外处理。

## 八、扩展说明

### 仅使用单教师模式
如果只需要 IRT 或 SPT 其中之一：
- 仅 IRT：代码中 SPT 权重路径不存在时会自动降级
- 仅 SPT：设置 `λ_irt=0` 即可

### 适配其他天气类型
修改 `setup_dataset.py` 中的天气类型列表即可：
```python
for wt in ["fog", "haze", "rain", "snow", "你的新天气类型"]:
```

### 导出模型用于推理
```python
import torch
from ultralytics import YOLO
model = YOLO('distill_output/stage2_final.pt')
model.predict('test_image.jpg', save=True)
```
