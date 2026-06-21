"""
双教师知识蒸馏训练器 — 两阶段训练

训练流程:
  Stage 1 (warmup/adaptation):
    - 冻结: 学生 Backbone + SPT 教师 + IRT 教师
    - 训练: 学生 Neck + FCA + Head
    - 学习率: 较高 (~1e-3)
    - 目标: Neck 学习融合双教师特征

  Stage 2 (global fine-tuning):
    - 冻结: 仅 SPT 教师 + IRT 教师
    - 训练: 学生全部参数
    - 学习率: 降低 (~1e-4)
    - 目标: 全局微调解耦
"""
import os
import sys
import time
import logging
import traceback
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# ── OmniRestore 路径 ──
_OMNIDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnirestore")
if os.path.isdir(_OMNIDIR):
    sys.path.insert(0, _OMNIDIR)

# ── 项目模块 ──────────────────────────────────────────────────────────
from aod_net import IRTTeacher, create_irt_teacher, create_omnirestore_teacher
from spt_teacher import SPTTeacher, create_spt_teacher
from losses import DualTeacherDistillLoss
from dataset import create_dataloaders

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# 工具函数
# ==============================================================================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str="0"):
    if device_str == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_str}")


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ==============================================================================
# YOLO Neck 特征提取器 (学生网络用)
# ==============================================================================
class YOLOFeatureExtractor:
    """通过钩子捕获 YOLO 模型中间层特征 (不额外运行前向)"""

    def __init__(self, model, layer_indices=(15, 22, 25)):
        self.model = model
        self.layer_indices = layer_indices
        self.features = {}
        self.handles = []

        for idx in layer_indices:
            if idx < len(model.model):
                handle = model.model[idx].register_forward_hook(
                    lambda m, i, o, _idx=idx: self.features.update({_idx: o.detach()})
                )
                self.handles.append(handle)

    def get_features(self):
        """返回最近一次前向传播缓存的特征 (不重新运行模型)"""
        feats = []
        for idx in sorted(self.layer_indices):
            if idx in self.features and self.features[idx] is not None:
                feats.append(self.features[idx])
        return feats if len(feats) == len(self.layer_indices) else []

    def remove(self):
        for h in self.handles:
            h.remove()


# ==============================================================================
# 层冻结工具
# ==============================================================================
def freeze_backbone(model):
    """冻结 YOLO backbone (layers 0-9)"""
    for i in range(10):
        if i < len(model.model):
            for p in model.model[i].parameters():
                p.requires_grad = False
    logger.info("[Freeze] Backbone (layers 0-9) 已冻结")


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True
    logger.info("[Unfreeze] 学生网络全部参数已解冻")


# ==============================================================================
# 双教师蒸馏训练器
# ==============================================================================
class DualTeacherDistillTrainer:
    """
    双教师知识蒸馏训练器

    前向传播逻辑:
      1. 天气原图 → IRT 教师 (OmniRestore/AOD-Net) → 复原图 + 边缘特征
      2. 天气原图 → SPT 教师 (YOLO-FCA) → Neck 语义特征
      3. 天气原图 → 学生 YOLO-FCA → 检测结果 + Neck 特征
      4. 计算 Det_Loss + IRT_Distill + SPT_Distill
    """

    def __init__(self, config):
        """
        config dict:
            spt_weights_path:     SPT 教师权重路径 (best.pt)
            irt_pretrained_path:  IRT 预训练权重 (可选)
            student_yaml:         学生 YOLO 配置文件
            student_pretrained:   学生预训练权重 (yolov8n.pt)
            data_root:            数据集根目录
            data_yaml:            YOLO 数据配置 yaml
            device:               设备
            img_size:             图像尺寸
            batch_size:           Batch 大小
            workers:              数据加载线程数
            stage1_epochs:        第一阶段训练轮数
            stage2_epochs:        第二阶段训练轮数
            stage1_lr:            第一阶段学习率
            stage2_lr:            第二阶段学习率
            lambda_irt:           IRT 蒸馏权重
            lambda_spt:           SPT 蒸馏权重
            save_dir:             模型保存目录
        """
        self.cfg = config
        self.device = get_device(config.get("device", "0"))
        set_seed(42)

        logger.info("=" * 60)
        logger.info("双教师知识蒸馏训练器初始化")
        logger.info("=" * 60)

        # ── 1. 创建教师 ──
        self._build_teachers()

        # ── 2. 创建学生 ──
        self._build_student()

        # ── 3. 损失函数 ──
        self._build_losses()

        # ── 4. 数据加载器 ──
        self._build_dataloaders()

        # ── 5. 训练状态 ──
        self.current_epoch = 0
        self.best_map = 0.0
        self.save_dir = Path(config.get("save_dir", "./distill_output"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── 构建函数 ─────────────────────────────────────────────────────

    def _build_teachers(self):
        """初始化双教师"""
        # SPT 教师 — 预训练雾天 YOLO
        spt_path = self.cfg["spt_weights_path"]
        if os.path.exists(spt_path):
            self.spt_teacher = create_spt_teacher(spt_path, device=str(self.device))
            self.spt_teacher = self.spt_teacher.to(self.device)
            logger.info(f"[Init] SPT 教师: {spt_path}")
        else:
            logger.warning(f"[Init] SPT 权重不存在: {spt_path}，将在第一阶段使用降级模式")

        # IRT 教师 — 默认 OmniRestore, 可回退到 AOD-Net
        use_omnirestore = self.cfg.get("use_omnirestore", True)
        if use_omnirestore:
            self.irt_teacher = create_omnirestore_teacher(
                output_channels=256,
                device=str(self.device),
                ckpt_path=self.cfg.get("omnirestore_ckpt"),
                embedder_path=self.cfg.get("omnirestore_embedder"),
            )
            logger.info("[Init] IRT 教师: OmniRestore (WADNet + CLIP-KAN), 已冻结")
        else:
            irt_pretrained = self.cfg.get("irt_pretrained_path", None)
            self.irt_teacher = create_irt_teacher(
                output_channels=256, pretrained_path=irt_pretrained,
                use_omnirestore=False,
            )
            self.irt_teacher = self.irt_teacher.to(self.device)
            logger.info("[Init] IRT 教师: AOD-Net + Sobel, 已冻结")

    def _build_student(self):
        """创建学生 YOLO-FCA 模型"""
        from ultralytics import YOLO
        student_yaml = self.cfg["student_yaml"]
        pretrained = self.cfg.get("student_pretrained", None)

        logger.info(f"[Init] 学生模型 yaml: {student_yaml}")
        if pretrained and os.path.exists(pretrained):
            self.student = YOLO(student_yaml, task="detect").load(pretrained)
            logger.info(f"[Init] 学生预训练权重: {pretrained}")
        else:
            self.student = YOLO(student_yaml, task="detect")
            logger.info("[Init] 学生模型从零初始化")

        self.student_model = self.student.model
        self.student_model.to(self.device)

        # 特征提取器 (用于蒸馏)
        self.student_extractor = YOLOFeatureExtractor(
            self.student_model, layer_indices=(15, 22, 25)
        )
        logger.info(f"[Init] 学生参数: {count_params(self.student_model):,}")

    def _build_losses(self):
        """初始化损失函数"""
        self.distill_loss_fn = DualTeacherDistillLoss(
            lambda_irt=self.cfg.get("lambda_irt", 0.3),
            lambda_spt=self.cfg.get("lambda_spt", 1.0),
            student_channels=[64, 128, 256],
            spt_channels=[64, 128, 256],
            irt_channels=256,
            aligned_ch=128,
        ).to(self.device)
        # 检测损失 (YOLO 内部计算)
        from ultralytics.utils.loss import v8DetectionLoss
        self.det_loss_fn = v8DetectionLoss(self.student_model).to(self.device)

    def _build_dataloaders(self):
        """创建数据加载器"""
        data_root = self.cfg["data_root"]
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=data_root,
            img_size=self.cfg.get("img_size", 640),
            batch_size=self.cfg.get("batch_size", 8),
            workers=self.cfg.get("workers", 4),
        )

    # ── 训练阶段 ─────────────────────────────────────────────────────

    def _run_epoch(self, loader, optimizer=None, scaler=None, desc="Train"):
        """运行一个 epoch"""
        is_train = optimizer is not None
        if is_train:
            self.student_model.train()
        else:
            self.student_model.eval()

        self.irt_teacher.eval()
        if hasattr(self, "spt_teacher"):
            self.spt_teacher.eval()

        total_loss = 0.0
        total_det = 0.0
        total_irt = 0.0
        total_spt = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=desc)
        for batch in pbar:
            imgs = batch["img"].to(self.device)  # [B, 3, 640, 640]

            # ── IRT 教师前向 (梯度关闭) ──
            with torch.no_grad():
                restored, irt_edge = self.irt_teacher(imgs)

            # ── SPT 教师前向 (梯度关闭) ──
            spt_feats = None
            if hasattr(self, "spt_teacher"):
                with torch.no_grad():
                    spt_feats = self.spt_teacher.get_neck_features(imgs)

            # ── 学生前向 + 检测损失 ──
            if is_train:
                # 标准 YOLO 训练前向
                student_out = self.student_model(imgs)
                # 获取学生 Neck 特征 (用于蒸馏)
                student_feats = self.student_extractor.get_features()

                # 检测损失 (使用 ultralytics 内置)
                # 需要构建目标格式
                targets = self._build_targets(batch, student_out)
                det_loss, det_loss_items = self.det_loss_fn(student_out, targets)
            else:
                with torch.no_grad():
                    student_out = self.student_model(imgs)
                    student_feats = self.student_extractor.get_features()
                    targets = self._build_targets(batch, student_out)
                    det_loss, det_loss_items = self.det_loss_fn(student_out, targets)

            # ── 蒸馏损失 ──
            if spt_feats and len(student_feats) >= 3 and irt_edge is not None:
                total, loss_info = self.distill_loss_fn(
                    det_loss, student_feats, irt_edge, spt_feats
                )
            else:
                total = det_loss
                loss_info = {"det_loss": det_loss.item(), "irt_distill": 0, "spt_distill": 0}

            # ── 反向传播 ──
            if is_train and optimizer is not None:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    optimizer.step()

            # 统计
            total_loss += total.item() if isinstance(total, torch.Tensor) else total
            total_det += loss_info["det_loss"]
            total_irt += loss_info["irt_distill"]
            total_spt += loss_info["spt_distill"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.3f}",
                "det": f"{total_det / num_batches:.3f}",
                "irt": f"{total_irt / num_batches:.4f}",
                "spt": f"{total_spt / num_batches:.4f}",
            })

        num_batches = max(num_batches, 1)
        return {
            "loss": total_loss / num_batches,
            "det_loss": total_det / num_batches,
            "irt_distill": total_irt / num_batches,
            "spt_distill": total_spt / num_batches,
        }

    def _build_targets(self, batch, predictions):
        """构建 YOLO v8DetectionLoss 需要的目标格式"""
        targets = torch.cat([
            torch.cat([
                torch.full((len(lbl), 1), i, device=self.device),
                lbl.to(self.device)
            ], dim=1)
            for i, lbl in enumerate(batch["labels"]) if len(lbl) > 0
        ], dim=0) if any(len(lbl) > 0 for lbl in batch["labels"]) else \
            torch.zeros((0, 6), device=self.device)
        return targets

    # ── 公开训练接口 ─────────────────────────────────────────────────

    def _run_map_eval(self, desc="Val"):
        """使用 ultralytics 内置 val() 计算 mAP"""
        if not hasattr(self, "student") or self.student is None:
            return {}
        try:
            data_yaml = self.cfg.get("data_yaml")
            if not data_yaml or not os.path.exists(data_yaml):
                logger.warning(f"[mAP] data_yaml 不存在: {data_yaml}")
                return {}

            results = self.student.val(
                data=data_yaml,
                imgsz=self.cfg.get("img_size", 640),
                batch=self.cfg.get("batch_size", 8),
                device=self.device,
                split="val",          # 使用 val 集
                plots=False,
                save_json=False,
                verbose=False,
            )

            # 提取 mAP (兼容不同 ultralytics 版本)
            if hasattr(results, "box"):
                ap50 = float(getattr(results.box, "map50", 0))
                ap75 = float(getattr(results.box, "map75", 0))
                ap50_95 = float(getattr(results.box, "map", 0))
            else:
                ap50 = float(getattr(results, "box_map50", 0))
                ap75 = float(getattr(results, "box_map75", 0))
                ap50_95 = float(getattr(results, "box_map", 0))

            metrics = {
                "mAP50": float(ap50),
                "mAP75": float(ap75),
                "mAP50_95": float(ap50_95),
            }
            logger.info(f"[{desc}] mAP50={ap50:.4f}, mAP75={ap75:.4f}, mAP50:95={ap50_95:.4f}")
            return metrics

        except Exception as e:
            logger.warning(f"[mAP] 评估失败: {e}")
            return {}

    def train_stage1(self):
        """第一阶段: 冻结 backbone + 教师，训练 neck + FCA + head"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Neck/Head Adaptation")
        logger.info("=" * 60)

        # 冻结 backbone
        freeze_backbone(self.student_model)
        logger.info(f"可训练参数: {count_params(self.student_model, trainable_only=True):,}")

        # 优化器 (包含蒸馏损失投影层)
        distill_params = list(self.distill_loss_fn.parameters())
        student_params = [p for p in self.student_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(distill_params + student_params,
                                lr=self.cfg.get("stage1_lr", 1e-3), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.get("stage1_epochs", 30)
        )
        scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        epochs = self.cfg.get("stage1_epochs", 30)
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self._run_epoch(self.train_loader, optimizer, scaler,
                                            desc=f"S1 Train [{epoch}/{epochs}]")
            scheduler.step()

            # 验证 (loss + mAP)
            val_metrics = {}
            if self.val_loader and epoch % 5 == 0:
                val_metrics = self._run_epoch(self.val_loader, desc=f"S1 Val [{epoch}]")
                map_metrics = self._run_map_eval(desc=f"S1 Val [{epoch}]")
                if map_metrics and map_metrics.get("mAP50_95", 0) > self.best_map:
                    self.best_map = map_metrics["mAP50_95"]
                    best_ckpt = self.save_dir / "stage1_best_map.pt"
                    torch.save(self.student_model.state_dict(), best_ckpt)
                    logger.info(f"[S1] 最佳 mAP 模型保存: {best_ckpt} ({self.best_map:.4f})")

            # 日志
            lr = optimizer.param_groups[0]["lr"]
            map_str = ""
            if val_metrics:
                map_str = f" | mAP50={map_metrics.get('mAP50', 0):.4f}" if map_metrics else ""
            logger.info(f"S1 Epoch {epoch:3d}/{epochs} | LR {lr:.2e} | "
                        f"Loss {train_metrics['loss']:.4f} | "
                        f"Det {train_metrics['det_loss']:.4f} | "
                        f"IRT {train_metrics['irt_distill']:.4f} | "
                        f"SPT {train_metrics['spt_distill']:.4f}"
                        f"{map_str}")

            # 保存检查点
            if epoch % 10 == 0 or epoch == epochs:
                ckpt_path = self.save_dir / f"stage1_epoch{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "stage": 1,
                    "model_state_dict": self.student_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": train_metrics,
                }, ckpt_path)
                logger.info(f"[S1] 检查点保存: {ckpt_path}")

        # 保存第一阶段最终模型
        final_path = self.save_dir / "stage1_final.pt"
        torch.save(self.student_model.state_dict(), final_path)
        logger.info(f"[S1] Stage 1 完成, 最终模型: {final_path}")

    def train_stage2(self):
        """第二阶段: 解冻全部学生参数，全局微调"""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Global Fine-Tuning")
        logger.info("=" * 60)

        # 解冻全部参数
        unfreeze_all(self.student_model)
        logger.info(f"可训练参数: {count_params(self.student_model, trainable_only=True):,}")

        # 优化器 (包含蒸馏损失投影层)
        optimizer = optim.AdamW(
            list(self.distill_loss_fn.parameters()) + list(self.student_model.parameters()),
            lr=self.cfg.get("stage2_lr", 1e-4), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.get("stage2_epochs", 50)
        )
        scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        epochs = self.cfg.get("stage2_epochs", 50)
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self._run_epoch(self.train_loader, optimizer, scaler,
                                            desc=f"S2 Train [{epoch}/{epochs}]")
            scheduler.step()

            val_metrics = {}
            if self.val_loader and epoch % 5 == 0:
                val_metrics = self._run_epoch(self.val_loader, desc=f"S2 Val [{epoch}]")
                map_metrics = self._run_map_eval(desc=f"S2 Val [{epoch}]")
                if map_metrics and map_metrics.get("mAP50_95", 0) > self.best_map:
                    self.best_map = map_metrics["mAP50_95"]
                    best_ckpt = self.save_dir / "stage2_best_map.pt"
                    torch.save(self.student_model.state_dict(), best_ckpt)
                    logger.info(f"[S2] 最佳 mAP 模型保存: {best_ckpt} ({self.best_map:.4f})")

            lr = optimizer.param_groups[0]["lr"]
            map_str = ""
            if val_metrics:
                map_str = f" | mAP50={map_metrics.get('mAP50', 0):.4f}" if map_metrics else ""
            logger.info(f"S2 Epoch {epoch:3d}/{epochs} | LR {lr:.2e} | "
                        f"Loss {train_metrics['loss']:.4f} | "
                        f"Det {train_metrics['det_loss']:.4f} | "
                        f"IRT {train_metrics['irt_distill']:.4f} | "
                        f"SPT {train_metrics['spt_distill']:.4f}"
                        f"{map_str}")

            if epoch % 10 == 0 or epoch == epochs:
                ckpt_path = self.save_dir / f"stage2_epoch{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "stage": 2,
                    "model_state_dict": self.student_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": train_metrics,
                }, ckpt_path)
                logger.info(f"[S2] 检查点保存: {ckpt_path}")

        # 保存最终模型 (导出为 ultralytics 格式)
        final_path = self.save_dir / "stage2_final.pt"
        self.student_model.eval()
        torch.save(self.student_model.state_dict(), final_path)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"训练完成! 最终模型: {final_path}")
        logger.info(f"{'=' * 60}")

    def run(self):
        """完整两阶段训练"""
        start_time = time.time()
        logger.info(f"训练开始, 设备: {self.device}")
        logger.info(f"Stage 1 epochs: {self.cfg.get('stage1_epochs', 30)}, "
                    f"LR: {self.cfg.get('stage1_lr', 1e-3)}")
        logger.info(f"Stage 2 epochs: {self.cfg.get('stage2_epochs', 50)}, "
                    f"LR: {self.cfg.get('stage2_lr', 1e-4)}")
        logger.info(f"λ_IRT: {self.cfg.get('lambda_irt', 0.3)}, "
                    f"λ_SPT: {self.cfg.get('lambda_spt', 1.0)}")

        self.train_stage1()
        self.train_stage2()

        elapsed = time.time() - start_time
        logger.info(f"总训练时间: {elapsed / 60:.1f} 分钟")

        # 清理钩子
        self.student_extractor.remove()
        if hasattr(self, "spt_teacher") and hasattr(self.spt_teacher, "remove_hooks"):
            self.spt_teacher.remove_hooks()


# ==============================================================================
# 测试
# ==============================================================================
if __name__ == "__main__":
    print("双教师蒸馏训练器模块")
    print("请通过 train_dual_teacher.py 启动训练")
