"""
蒸馏模块初始化
"""
from .aod_net import IRTTeacher, AODNet, SobelEdgeExtractor
from .spt_teacher import SPTTeacher
from .losses import DualTeacherDistillLoss
from .dataset import PairedWeatherDataset, create_dataloaders
from .trainer import DualTeacherDistillTrainer

__all__ = [
    "IRTTeacher",
    "AODNet",
    "SobelEdgeExtractor",
    "SPTTeacher",
    "DualTeacherDistillLoss",
    "PairedWeatherDataset",
    "create_dataloaders",
    "DualTeacherDistillTrainer",
]
