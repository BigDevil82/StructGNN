"""
消融实验配置模块

定义不同消融实验的配置，通过开关控制各个组件的启用/禁用
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class GNNBackbone(Enum):
    """GNN骨干网络类型"""

    GATv2 = "gatv2"  # 默认：图注意力网络v2
    GCN = "gcn"  # 图卷积网络
    GraphSAGE = "sage"  # GraphSAGE
    GIN = "gin"  # 图同构网络


class ConditioningMethod(Enum):
    """条件注入方法"""

    FILM = "film"  # 默认：Feature-wise Linear Modulation
    CONCAT_EARLY = "concat_early"  # 早期拼接（输入层）
    CONCAT_LATE = "concat_late"  # 后期拼接（解码器前）
    NONE = "none"  # 无条件（移除条件信息）


@dataclass
class AblationConfig:
    """
    消融实验配置

    通过设置不同的开关组合，可以生成不同的消融实验变体
    """

    # === 实验标识 ===
    name: str = "full_model"  # 实验名称，用于结果保存

    # === 模型架构 ===
    backbone: GNNBackbone = GNNBackbone.GATv2
    conditioning: ConditioningMethod = ConditioningMethod.FILM
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4  # 仅对GAT有效
    dropout: float = 0.2

    # === 损失函数开关 ===
    use_bce_loss: bool = True  # 分类损失
    use_mse_loss: bool = True  # 回归损失
    use_iou_loss: bool = True  # Vector IoU损失
    use_consistency_loss: bool = False  # 相邻房间一致性损失（实验证明无效，默认关闭）
    use_density_loss: bool = True  # 全局密度约束损失

    # === 损失权重 ===
    cls_weight: float = 1.0
    reg_weight: float = 2.0
    iou_weight: float = 2.0
    consistency_weight: float = 0.5  # 仅当 use_consistency_loss=True 时生效
    density_weight_real: float = 0.05
    density_weight_fake_max: float = 0.1

    # === 训练策略开关 ===
    use_dual_stream: bool = True  # 双流训练（真实+伪造条件）
    use_warmup: bool = True  # 密度损失预热
    warmup_epochs: int = 20  # 预热轮数
    use_weighted_sampling: bool = True  # 加权采样（类别平衡）

    # === 数据增广开关 ===
    use_augmentation: bool = True  # 几何增广
    augmentation_modes: List[str] = field(
        default_factory=lambda: ["none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270"]
    )

    # === 训练超参数 ===
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # === 交叉验证 ===
    n_folds: int = 5

    def __post_init__(self):
        """验证配置的合法性"""
        if self.conditioning == ConditioningMethod.NONE and self.use_dual_stream:
            print("⚠️ 警告：无条件模式下双流训练无效，自动禁用")
            self.use_dual_stream = False

        if self.conditioning == ConditioningMethod.NONE and self.use_density_loss:
            print("⚠️ 警告：无条件模式下密度损失无效，自动禁用")
            self.use_density_loss = False


# ============================================================
# 预定义的消融实验配置
# ============================================================

ABLATION_CONFIGS = {
    # 完整模型（基线）
    "full_model": AblationConfig(
        name="full_model",
    ),
    # === 条件注入方法消融 ===
    "wo_film_concat_early": AblationConfig(
        name="wo_film_concat_early",
        conditioning=ConditioningMethod.CONCAT_EARLY,
    ),
    "wo_film_concat_late": AblationConfig(
        name="wo_film_concat_late",
        conditioning=ConditioningMethod.CONCAT_LATE,
    ),
    "wo_conditioning": AblationConfig(
        name="wo_conditioning",
        conditioning=ConditioningMethod.NONE,
    ),
    # === 训练策略消融 ===
    "wo_dual_stream": AblationConfig(
        name="wo_dual_stream",
        use_dual_stream=False,
    ),
    "wo_warmup": AblationConfig(
        name="wo_warmup",
        use_warmup=False,
    ),
    # === 损失函数消融 ===
    "wo_iou_loss": AblationConfig(
        name="wo_iou_loss",
        use_iou_loss=False,
    ),
    "wo_consistency_loss": AblationConfig(
        name="with_consistency_loss",
        use_consistency_loss=True,  # 启用一致性损失（消融对比）
    ),
    "wo_density_loss": AblationConfig(
        name="wo_density_loss",
        use_density_loss=False,
    ),
    "wo_all_auxiliary_loss": AblationConfig(
        name="wo_all_auxiliary_loss",
        use_iou_loss=False,
        use_density_loss=False,
        # consistency_loss 默认已关闭
    ),
    # === 数据增广消融 ===
    "wo_augmentation": AblationConfig(
        name="wo_augmentation",
        use_augmentation=False,
        augmentation_modes=["none"],
    ),
    # === GNN骨干网络消融 ===
    "backbone_gcn": AblationConfig(
        name="backbone_gcn",
        backbone=GNNBackbone.GCN,
    ),
    "backbone_sage": AblationConfig(
        name="backbone_sage",
        backbone=GNNBackbone.GraphSAGE,
    ),
    "backbone_gin": AblationConfig(
        name="backbone_gin",
        backbone=GNNBackbone.GIN,
    ),
}


def get_ablation_config(name: str) -> AblationConfig:
    """获取指定名称的消融配置"""
    if name not in ABLATION_CONFIGS:
        available = list(ABLATION_CONFIGS.keys())
        raise ValueError(f"未知的消融配置: {name}. 可用配置: {available}")
    return ABLATION_CONFIGS[name]


def list_ablation_configs() -> List[str]:
    """列出所有可用的消融配置"""
    return list(ABLATION_CONFIGS.keys())
