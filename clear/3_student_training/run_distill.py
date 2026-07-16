"""
双教师知识蒸馏 — 独立运行脚本（无缓存依赖）
直接运行此文件，10+10 轮，C盘不占用空间
"""
import os
import sys
import shutil

# === 第一步：强制清理所有 __pycache__ ===
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# === 第二步：环境变量 & Matplotlib 中文字体设置 ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = "D:/项目/tianqi/clear/.torch_cache"
os.environ["HF_HOME"] = "D:/项目/tianqi/clear/.hf_cache"
os.environ["ULTRALYTICS_VERBOSE"] = "False"
os.environ["MPLBACKEND"] = "Agg"
os.environ["COMET_DISABLED"] = "1"          # 禁止 comet 自动创建实验
os.environ["WANDB_DISABLED"] = "true"       # 禁止 wandb
os.environ["CLEARML_DISABLED"] = "true"     # 禁止 clearml

# 解决中文字体缺失导致的 UserWarning 警告
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 优先使用黑体或微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# === 第三步：路径 ===
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BASE_DIR)
_SPT_DIR = os.path.join(_PROJECT_DIR, "1_spt_training")
_IRT_DIR = os.path.join(_PROJECT_DIR, "2_irt")

sys.path.insert(0, os.path.join(_SPT_DIR, "ultralytics-main"))
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, _IRT_DIR)


def check():
    """检查配置文件是否存在"""
    ok = True
    files = {
        "SPT权重": os.path.join(_SPT_DIR, "runs", "fca_detect", "train_fca_cl", "weights", "best.pt"),
        "学生yaml": os.path.join(_SPT_DIR, "ultralytics-main", "ultralytics", "cfg", "models", "v8", "yolov8-fca.yaml"),
        "学生预训练": os.path.join(_BASE_DIR, "yolov8n.pt"),
        "数据集": os.path.join(_BASE_DIR, "data", "images", "train"),
        "OmniRestore权重": os.path.join(_IRT_DIR, "omnirestore", "ckpts", "best.ckpt"),
        "OmniRestore编码器": os.path.join(_IRT_DIR, "omnirestore", "logs", "embedder.pt"),
    }
    for name, path in files.items():
        status = "[OK]" if os.path.exists(path) else "[缺失]"
        print(f"  {status} {name}: {path}")
        if not os.path.exists(path):
            ok = False
    return ok


# ⚠️ Windows multiprocessing 要求主代码必须放在 __main__ 保护下
# DataLoader(num_workers>0) 会用 spawn 创建子进程，子进程会重新导入本模块
if __name__ == "__main__":
    print("=" * 70)
    print("双教师知识蒸馏训练")
    print("=" * 70)
    print("\n环境检查:")
    if not check():
        print("\n请修复以上问题后重试")
        sys.exit(1)

    # === 第五步：导入并运行 ===
    import importlib.util
    spec = importlib.util.spec_from_file_location("trainer", os.path.join(_BASE_DIR, "trainer.py"))
    trainer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_mod)

    cfg = {
        "spt_weights_path": os.path.join(_SPT_DIR, "runs", "fca_detect", "train_fca_cl", "weights", "best.pt"),
        "irt_pretrained_path": os.path.join(_IRT_DIR, "dehazer.pth"),
        "omnirestore_ckpt": os.path.join(_IRT_DIR, "omnirestore", "ckpts", "best.ckpt"),
        "omnirestore_embedder": os.path.join(_IRT_DIR, "omnirestore", "logs", "embedder.pt"),
        "use_omnirestore": True,
        "student_yaml": os.path.join(_SPT_DIR, "ultralytics-main", "ultralytics", "cfg", "models", "v8", "yolov8-fca.yaml"),
        "student_pretrained": os.path.join(_BASE_DIR, "yolov8n.pt"),
        "data_root": os.path.join(_BASE_DIR, "data"),
        "data_yaml": os.path.join(_BASE_DIR, "data_weather.yaml"),
        "device": "cpu",
        "img_size": 640,
        "batch_size": 8,
        "workers": 0,  # 设为 0 避免 Windows 上 multiprocessing 子进程的解码兼容问题

        # 真正的两阶段轮数与学习率控制
        "stage1_epochs": 2,
        "stage2_epochs": 2,
        "stage1_lr": 1e-3,
        "stage2_lr": 1e-4,

        "lambda_irt": 0.3,
        "lambda_spt": 1.0,
        "save_dir": os.path.join(_BASE_DIR, "distill_output"),
    }

    # 禁用 ultralytics 的集成回调（comet/wandb 等），防止原生训练器被误触
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS["comet"] = False
        SETTINGS["wandb"] = False
        SETTINGS["clearml"] = False
        SETTINGS["mlflow"] = False
        SETTINGS["neptune"] = False
        print("  [OK] 已禁用 ultralytics 集成回调")
    except Exception:
        pass

    print(f"\n开始训练 (Stage1: {cfg['stage1_epochs']}轮 + Stage2: {cfg['stage2_epochs']}轮)...")
    print(f"  批次大小: {cfg['batch_size']}, 设备: {cfg['device']}, "
          f"工作线程: {cfg['workers']}")
    print(f"  缓存目录: TORCH_HOME={os.environ.get('TORCH_HOME', '默认(C盘)')}")
    print(f"  缓存目录: HF_HOME={os.environ.get('HF_HOME', '默认(C盘)')}")

    trainer = trainer_mod.DualTeacherDistillTrainer(cfg)
    trainer.run()
    print("\n训练完成!")