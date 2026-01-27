"""
消融实验模型变体

实现不同的模型架构变体用于消融研究：
1. 不同的GNN骨干网络 (GATv2, GCN, GraphSAGE, GIN)
2. 不同的条件注入方法 (FiLM, Concat-Early, Concat-Late, None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
)

from experiments.ablation.config import AblationConfig, ConditioningMethod, GNNBackbone


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation层"""

    def __init__(self, in_dim: int, cond_emb_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_emb_dim, in_dim)
        self.shift = nn.Linear(cond_emb_dim, in_dim)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(cond_emb)
        beta = self.shift(cond_emb)
        return x * (1 + gamma) + beta


class AblationShearWallGNN(nn.Module):
    """
    可配置的剪力墙预测GNN模型

    支持不同的骨干网络和条件注入方法，用于消融实验
    """

    def __init__(self, config: AblationConfig, node_in_dim: int = 25, edge_in_dim: int = 8, cond_dim: int = 3):
        super().__init__()

        self.config = config
        self.backbone_type = config.backbone
        self.conditioning_type = config.conditioning
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.cond_dim = cond_dim
        self.cond_emb_dim = 32

        # === 条件编码器 ===
        if self.conditioning_type != ConditioningMethod.NONE:
            self.cond_projector = nn.Sequential(
                nn.Linear(cond_dim, self.cond_emb_dim),
                nn.ReLU(),
                nn.Linear(self.cond_emb_dim, self.cond_emb_dim),
            )

        # === 输入编码器 ===
        # 根据条件注入方法调整输入维度
        if self.conditioning_type == ConditioningMethod.CONCAT_EARLY:
            actual_node_in_dim = node_in_dim + self.cond_emb_dim
        else:
            actual_node_in_dim = node_in_dim

        self.node_encoder = nn.Linear(actual_node_in_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, self.hidden_dim)

        # === GNN层 ===
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.num_layers):
            conv = self._create_conv_layer(i)
            self.convs.append(conv)
            self.bns.append(BatchNorm(self.hidden_dim))

        # === FiLM层（仅当使用FiLM条件注入时） ===
        if self.conditioning_type == ConditioningMethod.FILM:
            self.film_layers = nn.ModuleList([
                FiLMLayer(self.hidden_dim, self.cond_emb_dim)
                for _ in range(self.num_layers)
            ])

        # === 解码器 ===
        if self.conditioning_type == ConditioningMethod.CONCAT_LATE:
            decoder_in_dim = self.hidden_dim + self.cond_emb_dim
        else:
            decoder_in_dim = self.hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 32),  # 16 cls + 16 reg
        )

    def _create_conv_layer(self, layer_idx: int) -> nn.Module:
        """根据配置创建对应的GNN卷积层"""

        if self.backbone_type == GNNBackbone.GATv2:
            return GATv2Conv(
                self.hidden_dim,
                self.hidden_dim,
                heads=self.config.num_heads,
                concat=False,
                edge_dim=self.hidden_dim,
            )

        elif self.backbone_type == GNNBackbone.GCN:
            return GCNConv(self.hidden_dim, self.hidden_dim)

        elif self.backbone_type == GNNBackbone.GraphSAGE:
            return SAGEConv(self.hidden_dim, self.hidden_dim)

        elif self.backbone_type == GNNBackbone.GIN:
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            return GINConv(mlp)

        else:
            raise ValueError(f"未知的骨干网络类型: {self.backbone_type}")

    def forward(
        self, data: Data, condition: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            data: PyG Data对象
            condition: 可选的条件向量，用于覆盖data中的条件

        Returns:
            (classification_prob, regression_ratio)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        # === 条件编码 ===
        if self.conditioning_type != ConditioningMethod.NONE:
            raw_condition = condition if condition is not None else data.condition
            cond_emb = self.cond_projector(raw_condition)  # (Batch, cond_emb_dim)
            node_cond_emb = cond_emb[batch]  # (N, cond_emb_dim)

        # === 早期拼接 ===
        if self.conditioning_type == ConditioningMethod.CONCAT_EARLY:
            x = torch.cat([x, node_cond_emb], dim=-1)

        # === 特征编码 ===
        x = F.relu(self.node_encoder(x))
        edge_attr_encoded = F.relu(self.edge_encoder(edge_attr))

        # === GNN层（带残差连接） ===
        for i in range(self.num_layers):
            x_residual = x

            # 不同backbone的forward调用方式不同
            if self.backbone_type == GNNBackbone.GATv2:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr_encoded)
            elif self.backbone_type in [GNNBackbone.GCN, GNNBackbone.GraphSAGE, GNNBackbone.GIN]:
                x = self.convs[i](x, edge_index)

            x = self.bns[i](x)

            # FiLM调制
            if self.conditioning_type == ConditioningMethod.FILM:
                x = self.film_layers[i](x, node_cond_emb)

            x = F.relu(x) + x_residual

        # === 后期拼接 ===
        if self.conditioning_type == ConditioningMethod.CONCAT_LATE:
            x = torch.cat([x, node_cond_emb], dim=-1)

        # === 解码输出 ===
        out = self.decoder(x)

        classification_logits = out[:, :16]
        regression_logits = out[:, 16:]

        classification_prob = torch.sigmoid(classification_logits)
        regression_ratio = torch.sigmoid(regression_logits)

        return classification_prob, regression_ratio


def create_model_from_config(config: AblationConfig, node_in_dim: int = 25, edge_in_dim: int = 8) -> nn.Module:
    """
    根据消融配置创建模型

    Args:
        config: 消融实验配置
        node_in_dim: 节点特征维度
        edge_in_dim: 边特征维度

    Returns:
        配置好的模型实例
    """
    return AblationShearWallGNN(
        config=config,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
    )
