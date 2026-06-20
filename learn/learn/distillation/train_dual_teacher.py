"""
双教师知识蒸馏恶劣天气目标检测 — 主训练入口

使用方法:
    python train_dual_teacher.py

运行前:
    1. 运行 setup_dataset.py 创建 8:1:1 数据集
    2. 确保 SPT 教师权重 (best.pt) 已存在
    3. 确保 ultralytics-main 在 sys.path 中

架构:
    SPT (语义教师, 冻结) ──→ Neck 语义特征 ──┐
                                              ├──→ 学生 YOLO-FCA
    IRT (重建教师, 冻结) ──→ 边缘轮廓特征 ──┘

训练:
    Stage 1: 冻结 Backbone, 训练 Neck/FCA/Head, LR=1e-3, 30 epochs
    Stage 2: 解冻全部, 全局微调, LR=1e-4, 50 epochs
"""
import os
import sys

# ── 路径设置 ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BASE_DIR)  # learn\learn\

# 添加 ultralytics-main 到 Python 路径
sys.path.insert(0, os.path.join(_PROJECT_DIR, "ultralytics-main"))
# 添加当前 distillation 目录
sys.path.insert(0, _BASE_DIR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from trainer import DualTeacherDistillTrainer


# ==============================================================================
# 配置
# ==============================================================================
def get_default_config():
    """默认训练配置 — 根据实际路径调整"""
    return {
        # ── 路径 ──
        # SPT 教师: 预训练雾天 YOLO-FCA 模型 (先用 train.py 训练得到)
        "spt_weights_path": os.path.join(
            _PROJECT_DIR, "runs", "fca_detect", "train_fca_cl", "weights", "best.pt"
        ),
        # IRT 教师: AOD-Net 预训练权重 (可选，None 则随机初始化)
        "irt_pretrained_path": None,
        # 学生模型: YOLOv8-FCA yaml 配置
        "student_yaml": os.path.join(
            _PROJECT_DIR, "ultralytics-main", "ultralytics", "cfg",
            "models", "v8", "yolov8-fca.yaml"
        ),
        # 学生预训练权重
        "student_pretrained": os.path.join(_PROJECT_DIR, "源代码文件夹", "yolov8n.pt"),
        # 数据集根目录 (setup_dataset.py 生成)
        "data_root": os.path.join(_PROJECT_DIR, "..", "..", "adverse_weather_yolo"),
        # 数据 yaml (用于 ultralytics 内置评估)
        "data_yaml": os.path.join(_BASE_DIR, "data_weather.yaml"),

        # ── 设备 ──
        "device": "0",  # GPU 设备号, "cpu" 表示 CPU
        "img_size": 640,
        "batch_size": 8,
        "workers": 4,

        # ── Stage 1 超参数 (Neck/Head Adaptation) ──
        "stage1_epochs": 30,
        "stage1_lr": 1e-3,
        # ── Stage 2 超参数 (Global Fine-tuning) ──
        "stage2_epochs": 50,
        "stage2_lr": 1e-4,

        # ── 蒸馏权重 ──
        # λ1: IRT 蒸馏权重, 范围建议 [0.1, 1.0]
        #   值越大, 学生越依赖 IRT 的边缘/重建特征
        #   如果学生检测结果中边缘信息不足, 可增大
        "lambda_irt": 0.3,
        # λ2: SPT 蒸馏权重, 范围建议 [0.5, 2.0]
        #   值越大, 学生越模仿 SPT 的语义特征
        #   如果学生检测精度不够, 可增大
        "lambda_spt": 1.0,

        # ── 保存 ──
        "save_dir": os.path.join(_BASE_DIR, "distill_output"),
    }


def validate_config(cfg):
    """验证配置完整性, 输出缺失项"""
    issues = []
    if not os.path.exists(cfg["student_yaml"]):
        issues.append(f"学生模型 yaml 不存在: {cfg['student_yaml']}")
    if not os.path.exists(cfg["student_pretrained"]):
        issues.append(f"学生预训练权重不存在: {cfg['student_pretrained']} (将随机初始化)")
    if not os.path.exists(cfg["data_root"]):
        issues.append(f"数据集目录不存在: {cfg['data_root']} (请先运行 setup_dataset.py)")
    if not os.path.exists(cfg["spt_weights_path"]):
        issues.append(f"SPT 教师权重不存在: {cfg['spt_weights_path']} (请先训练基础 YOLO-FCA 模型)")

    if issues:
        print("\n⚠️  配置检查发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        return False
    return True


# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("双教师知识蒸馏 — 恶劣天气目标检测训练")
    print("=" * 70)

    cfg = get_default_config()

    if not validate_config(cfg):
        print("请修正以上问题后重新运行。")
        print("=" * 70)
        sys.exit(1)

    print("\n配置摘要:")
    for k, v in cfg.items():
        if isinstance(v, str) and os.path.exists(v):
            status = "✅"
        elif isinstance(v, str):
            status = "⚠️"
        else:
            status = "  "
        print(f"  {status} {k}: {v}")

    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    trainer = DualTeacherDistillTrainer(cfg)
    trainer.run()
