"""
训练和推理的配置参数
集中管理所有硬编码的常量和超参数
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch


@dataclass
class ModelConfig:
    """模型架构配置"""

    # 输入特征维度
    NODE_FEATURE_DIM: int = 25  # 几何特征(9) + 约束特征(16)
    GEO_FEATURE_DIM: int = 9  # 几何特征维度
    CONSTRAINT_DIM: int = 16  # 约束/标签维度 (4边 × 2半边 × 2端点)
    EDGE_FEATURE_DIM: int = 8  # 边特征维度

    # 模型超参数
    HIDDEN_DIM: int = 256
    OUTPUT_DIM: int = 32  # 分类(16) + 回归(16)
    GAT_HEADS: int = 4
    NUM_CONV_LAYERS: int = 3
    DROPOUT: float = 0.2


@dataclass
class TrainingConfig:
    """训练配置"""

    # 训练参数
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 100

    # 数据集划分比例
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.2
    TEST_RATIO: float = 0.1

    # Loss权重
    CLS_WEIGHT: float = 1.0  # 分类损失权重
    REG_WEIGHT: float = 2.0  # 回归损失权重
    CONSISTENCY_WEIGHT: float = 0.5  # 一致性损失权重
    MASK_PENALTY_WEIGHT: float = 5.0  # 物理约束惩罚权重

    # 设备
    DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机种子
    RANDOM_SEED: int = 42


@dataclass
class DataConfig:
    """数据处理配置"""

    # 路径
    DXF_DIR: str = r"dxf/dataset_split_8_2"
    CACHE_DIR: str = "data_cache"
    SAVE_DIR: str = "result/1226_cond_kfold"

    # 数据增广
    AUGMENTATIONS: List[str] = (
        "none",
        "flip_x",
        "flip_y",
        "rot_90",
        "rot_180",
        "rot_270",
    )  # 支持的增广模式

    # 预处理参数
    ALIGNMENT_THRESHOLD: float = 200.0  # 房间校准的坐标对齐阈值
    MIN_SHARED_LENGTH: float = 400.0  # 边连接的最小共享长度

    # 区间合并阈值
    INTERVAL_THRESHOLD: float = 0.1  # 墙体区间的最小长度阈值

    # 可布置区域判定阈值
    BUILDABLE_THRESHOLD: float = 0.4  # 低于此比例视为不可布置


@dataclass
class VisualizationConfig:
    """可视化配置"""

    # 预测阈值
    PRED_PROB_THRESHOLD: float = 0.5  # 分类概率阈值
    PRED_RATIO_THRESHOLD: float = 0.1  # 回归比例阈值（低于此值置0）

    # IoU计算
    IOU_BUFFER_WIDTH: float = 0.5  # 计算IoU时的buffer宽度

    # 绘图参数
    GT_WALL_COLOR: str = "green"  # Ground Truth墙体颜色
    PRED_WALL_COLOR: str = "red"  # 预测墙体颜色
    BUILDABLE_COLOR: str = "#BBBBBB"  # 可布置区域颜色

    # 输出
    DPI: int = 300
    FIG_SIZE_DOUBLE: tuple = (8, 6)


# 全局配置实例
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()
viz_config = VisualizationConfig()
