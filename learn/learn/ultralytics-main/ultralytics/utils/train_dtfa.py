
"""
DTFA Training Script
双教师框架训练脚本

使用方法：
    python train_dtfa.py --data your_dataset.yaml --epochs 200 --batch 8

创新点体现：
1. 位置一：AFB模块内的显式对比学习对齐
2. 位置二：特征级对比损失（权重0.1）
3. 位置三：IRT编码器中的FCA特征聚集
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # ultralytics根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.nn.modules.dtfa_framework import (
    IRTTeacher,
    SPTTeacher,
    DTFAStudentNetwork,
    DTFALoss
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER, colorstr


def parse_opt():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DTFA双教师框架训练')
    
    # 数据集参数
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=4, help='类别数量')
    parser.add_argument('--use-fca', action='store_true', default=True, help='是否使用FCA增强')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.937, help='动量')
    
    # 损失权重
    parser.add_argument('--lambda-recon', type=float, default=1.0, help='重建损失权重')
    parser.add_argument('--lambda-contrastive', type=float, default=0.1, help='对比损失权重')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--project', type=str, default='runs/dtfa', help='项目保存路径')
    parser.add_argument('--name', type=str, default='train', help='实验名称')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    return parser.parse_args()


class DTFATrainer:
    """DTFA双教师框架训练器"""
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(opt.device)
        
        # 创建保存目录
        self.save_dir = Path(opt.project) / opt.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(colorstr('bright_blue', 'bold', '\n初始化DTFA双教师框架'))
        
        # === 1. 初始化三个网络 ===
        LOGGER.info("创建IRT教师网络...")
        self.irt_teacher = IRTTeacher(
            in_channels=3,
            base_channels=64,
            use_fca=opt.use_fca  # 位置三的创新点
        ).to(self.device)
        
        LOGGER.info("创建SPT教师网络...")
        self.spt_teacher = SPTTeacher(
            num_classes=opt.num_classes,
            pretrained=True
        ).to(self.device)
        
        LOGGER.info("创建学生网络...")
        self.student = DTFAStudentNetwork(
            num_classes=opt.num_classes,
            use_fca_cl=opt.use_fca
        ).to(self.device)
        
        # === 2. 初始化损失函数 ===
        LOGGER.info("创建损失函数...")
        self.criterion = DTFALoss(
            lambda_recon=opt.lambda_recon,
            lambda_contrastive=opt.lambda_contrastive  # 位置二的创新点
        ).to(self.device)
        
        # === 3. 初始化优化器 ===
        # 只优化学生网络的参数，教师网络参数冻结或缓慢更新
        params_to_optimize = list(self.student.parameters())
        
        # IRT教师可以选择性微调
        for param in self.irt_teacher.parameters():
            param.requires_grad = False  # 初期冻结，后期可以解冻
        
        self.optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
        )
        
        # === 4. 学习率调度器 ===
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=opt.epochs,
            eta_min=opt.lr * 0.01
        )
        
        # === 5. 数据加载 ===
        LOGGER.info("加载数据集...")
        self.train_loader, self.val_loader = self._create_dataloaders(opt.data)
        
        # === 6. 训练记录 ===
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.training_stats = {
            'epoch': [],
            'total_loss': [],
            'det_loss': [],
            'recon_loss': [],
            'contrastive_loss': []
        }
        
        LOGGER.info(f"模型参数量:")
        LOGGER.info(f"  IRT教师: {sum(p.numel() for p in self.irt_teacher.parameters()):,}")
        LOGGER.info(f"  SPT教师: {sum(p.numel() for p in self.spt_teacher.parameters()):,}")
        LOGGER.info(f"  学生网络: {sum(p.numel() for p in self.student.parameters()):,}")
    
    def _create_dataloaders(self, data_yaml):
        """创建训练和验证数据加载器"""
        # 这里简化实现，实际应该根据YOLO数据集格式加载
        # 对于完整实现，需要使用ultralytics的数据集类
        
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_dataloader
        
        cfg = get_cfg(overrides={'data': data_yaml})
        
        # 训练集
        train_dataset = YOLODataset(
            img_path=cfg.train_path,
            labels=cfg.train_labels,
            augment=True,
            hyp=cfg,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=colorstr('train: ')
        )
        
        # 验证集
        val_dataset = YOLODataset(
            img_path=cfg.val_path,
            labels=cfg.val_labels,
            augment=False,
            hyp=cfg,
            rect=True,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix=colorstr('val: ')
        )
        
        train_loader = build_dataloader(
            dataset=train_dataset,
            batch=self.opt.batch,
            workers=self.opt.workers,
            shuffle=True,
            rank=-1
        )
        
        val_loader = build_dataloader(
            dataset=val_dataset,
            batch=self.opt.batch,
            workers=self.opt.workers,
            shuffle=False,
            rank=-1
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.student.train()
        self.irt_teacher.eval()
        self.spt_teacher.eval()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        epoch_losses = {
            'total': 0.0,
            'detection': 0.0,
            'reconstruction': 0.0,
            'feature_contrastive': 0.0,
            'afb_contrastive': 0.0
        }
        
        for i, batch in pbar:
            # 准备数据
            imgs = batch['img'].to(self.device)  # 退化图像
            targets = batch['cls'].to(self.device) if 'cls' in batch else None
            
            # 假设有对应的干净图像用于重建监督
            # 实际应用中需要从数据集中获取paired数据
            clean_imgs = imgs.clone()  # 这里简化，实际应该是干净的配对图像
            
            # === 前向传播 ===
            
            # 1. IRT教师：重建干净图像
            with torch.no_grad():
                recon_img, irt_features = self.irt_teacher(imgs)
                irt_feat = irt_features['enc3']  # 提取中间特征
            
            # 2. SPT教师：提供语义特征
            with torch.no_grad():
                spt_det, spt_features = self.spt_teacher(clean_imgs)
                spt_feat = spt_features['semantic']
            
            # 3. 学生网络：检测 + AFB对齐
            det_pred, afb_contrastive_loss, student_feat = self.student(
                imgs, irt_feat, spt_feat, targets
            )
            
            # === 计算损失 ===
            det_target = torch.zeros_like(det_pred)  # 简化，实际应该有真实标注
            total_loss, loss_dict = self.criterion(
                recon_img, clean_imgs,
                det_pred, det_target,
                student_feat, irt_feat, spt_feat,
                afb_contrastive_loss=afb_contrastive_loss
            )
            
            # === 反向传播 ===
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 累积损失
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            
            # 更新进度条
            pbar.set_description(
                f"Epoch {epoch}/{self.opt.epochs} | "
                f"Loss: {loss_dict['total'].item():.4f} | "
                f"Det: {loss_dict['detection'].item():.4f} | "
                f"Recon: {loss_dict['reconstruction'].item():.4f}"
            )
        
        # 平均损失
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self):
        """验证模型"""
        self.student.eval()
        
        val_losses = {
            'total': 0.0,
            'detection': 0.0,
            'reconstruction': 0.0
        }
        
        for batch in self.val_loader:
            imgs = batch['img'].to(self.device)
            clean_imgs = imgs.clone()
            
            # IRT重建
            recon_img, irt_features = self.irt_teacher(imgs)
            irt_feat = irt_features['enc3']
            
            # SPT语义
            spt_det, spt_features = self.spt_teacher(clean_imgs)
            spt_feat = spt_features['semantic']
            
            # 学生检测
            det_pred, _, student_feat = self.student(imgs, irt_feat, spt_feat)
            
            # 计算损失
            det_target = torch.zeros_like(det_pred)
            total_loss, loss_dict = self.criterion(
                recon_img, clean_imgs,
                det_pred, det_target,
                student_feat, irt_feat, spt_feat
            )
            
            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key].item()
        
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'irt_teacher_state_dict': self.irt_teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_stats': self.training_stats
        }
        
        # 保存最新检查点
        save_path = self.save_dir / 'last.pt'
        torch.save(checkpoint, save_path)
        LOGGER.info(f"保存检查点到: {save_path}")
        
        # 保存最佳检查点
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            LOGGER.info(colorstr('green', 'bold', f"保存最佳模型到: {best_path}"))
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        LOGGER.info(f"从 {checkpoint_path} 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.irt_teacher.load_state_dict(checkpoint['irt_teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.training_stats = checkpoint['training_stats']
        
        LOGGER.info(f"恢复到 epoch {self.start_epoch}")
    
    def train(self):
        """主训练循环"""
        LOGGER.info(colorstr('bright_yellow', 'bold', '\n开始DTFA训练'))
        LOGGER.info(f"训练参数:")
        LOGGER.info(f"  Epochs: {self.opt.epochs}")
        LOGGER.info(f"  Batch size: {self.opt.batch}")
        LOGGER.info(f"  Learning rate: {self.opt.lr}")
        LOGGER.info(f"  Lambda recon: {self.opt.lambda_recon}")
        LOGGER.info(f"  Lambda contrastive: {self.opt.lambda_contrastive}")
        
        # 恢复训练（如果需要）
        if self.opt.resume:
            self.load_checkpoint(self.opt.resume)
        
        # 训练循环
        for epoch in range(self.start_epoch, self.opt.epochs):
            LOGGER.info(colorstr('cyan', 'bold', f'\nEpoch {epoch+1}/{self.opt.epochs}'))
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % 5 == 0 or epoch == self.opt.epochs - 1:
                val_losses = self.validate()
                LOGGER.info(
                    f"验证结果 - "
                    f"Total: {val_losses['total']:.4f}, "
                    f"Det: {val_losses['detection']:.4f}, "
                    f"Recon: {val_losses['reconstruction']:.4f}"
                )
                
                # 保存最佳模型
                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.save_checkpoint(epoch, is_best=True)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录统计信息
            self.training_stats['epoch'].append(epoch)
            self.training_stats['total_loss'].append(train_losses['total'])
            self.training_stats['det_loss'].append(train_losses['detection'])
            self.training_stats['recon_loss'].append(train_losses['reconstruction'])
            self.training_stats['contrastive_loss'].append(train_losses['feature_contrastive'])
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        LOGGER.info(colorstr('green', 'bold', '\n训练完成！'))
        LOGGER.info(f"最佳验证损失: {self.best_loss:.4f}")
        LOGGER.info(f"模型保存在: {self.save_dir}")


def main():
    """主函数"""
    opt = parse_opt()
    
    # 打印参数
    LOGGER.info(colorstr('bright_blue', 'bold', '\nDTFA双教师框架训练配置'))
    LOGGER.info("=" * 80)
    for key, value in vars(opt).items():
        LOGGER.info(f"{key:30s}: {value}")
    LOGGER.info("=" * 80)
    
    # 创建训练器并开始训练
    trainer = DTFATrainer(opt)
    trainer.train()


if __name__ == '__main__':
    main()
