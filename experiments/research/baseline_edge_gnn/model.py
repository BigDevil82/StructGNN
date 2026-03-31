"""
GNN-EP-4 Model Architecture

Implementation of the edge prediction GNN model from:
"Intelligent design of shear wall layout based on graph neural networks"

Architecture:
- SFLayer backbone (Graph-SAGE style spatial GNN)
- Batch Normalization + Dropout
- MLP prediction head for edge features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data

from experiments.research.baseline_edge_gnn.config import model_config


class SFLayer(nn.Module):
    """
    Spatial Feature Layer based on GraphSAGE.

    This is the core GNN layer used in the paper's GNN-EP-4 model.
    Uses mean aggregation as in the Graph-SF variant.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SFLayer, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels, aggr="mean")

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))


class GNNBackbone(nn.Module):
    """
    GNN backbone with stacked SFLayers, BatchNorm, and Dropout.

    Architecture follows GNN-EP-4:
    - SFLayer(D_in, 16) -> SFLayer(16, 16) -> BN + Dropout
    - SFLayer(16, 32) -> SFLayer(32, 32) -> BN + Dropout
    - SFLayer(32, 32) -> SFLayer(32, D_out)
    """

    def __init__(
        self,
        in_dim: int = model_config.NODE_IN_DIM,
        hidden_dims: list = None,
        out_dim: int = model_config.GNN_OUT_DIM,
        dropout: float = model_config.DROPOUT,
    ):
        super(GNNBackbone, self).__init__()

        if hidden_dims is None:
            hidden_dims = model_config.HIDDEN_DIMS

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        # Build layers according to paper architecture
        prev_dim = in_dim

        # Block 1: SFLayer(in, 16) -> SFLayer(16, 16) -> BN + Dropout
        self.layers.append(SFLayer(prev_dim, 16))
        self.layers.append(SFLayer(16, 16))
        self.norms.append(BatchNorm(16))
        prev_dim = 16

        # Block 2: SFLayer(16, 32) -> SFLayer(32, 32) -> BN + Dropout
        self.layers.append(SFLayer(prev_dim, 32))
        self.layers.append(SFLayer(32, 32))
        self.norms.append(BatchNorm(32))
        prev_dim = 32

        # Block 3: SFLayer(32, 32) -> SFLayer(32, out_dim)
        self.layers.append(SFLayer(prev_dim, 32))
        self.layers.append(SFLayer(32, out_dim))

    def forward(self, x, edge_index):
        # Block 1
        x = self.layers[0](x, edge_index)
        x = self.layers[1](x, edge_index)
        x = self.norms[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Block 2
        x = self.layers[2](x, edge_index)
        x = self.layers[3](x, edge_index)
        x = self.norms[1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Block 3
        x = self.layers[4](x, edge_index)
        x = self.layers[5](x, edge_index)

        return x


class EdgeMLPHead(nn.Module):
    """
    MLP prediction head for edge features.

    Input: Concatenation of [edge_feature, src_node_embedding, dst_node_embedding]
    Output: [ratio_start, ratio_end] for each edge

    Architecture:
    - Linear(input, 32) -> Linear(32, 32) -> BN + Dropout
    - Linear(32, 16) -> Linear(16, 16) -> BN + Dropout
    - Linear(16, 8) -> Linear(8, 2)
    """

    def __init__(
        self,
        edge_dim: int = model_config.EDGE_IN_DIM,
        node_embed_dim: int = model_config.GNN_OUT_DIM,
        dropout: float = model_config.DROPOUT,
    ):
        super(EdgeMLPHead, self).__init__()

        # Input dimension: edge features + 2 * node embeddings
        input_dim = edge_dim + 2 * node_embed_dim

        # MLP layers
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)

        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 2)

        self.dropout = dropout

    def forward(self, edge_attr, src_embed, dst_embed):
        # Concatenate edge features with node embeddings
        x = torch.cat([edge_attr, src_embed, dst_embed], dim=-1)

        # Block 1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Block 2
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))  # Output in [0, 1] for ratios

        return x


class GNNEP4(nn.Module):
    """
    GNN-EP-4: Graph Neural Network for Edge Prediction (Model 4)

    Complete model combining:
    - GNN backbone for node embedding
    - MLP head for edge prediction

    Input:
        - x: Node features [num_nodes, 2] (normalized coordinates)
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Edge features [num_edges, 9]

    Output:
        - Edge predictions [num_edges, 2] (ratio_start, ratio_end)
    """

    def __init__(
        self,
        node_in_dim: int = model_config.NODE_IN_DIM,
        edge_in_dim: int = model_config.EDGE_IN_DIM,
        gnn_out_dim: int = model_config.GNN_OUT_DIM,
        dropout: float = model_config.DROPOUT,
    ):
        super(GNNEP4, self).__init__()

        self.backbone = GNNBackbone(
            in_dim=node_in_dim,
            out_dim=gnn_out_dim,
            dropout=dropout,
        )

        self.head = EdgeMLPHead(
            edge_dim=edge_in_dim,
            node_embed_dim=gnn_out_dim,
            dropout=dropout,
        )

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Get node embeddings from GNN backbone
        node_embed = self.backbone(x, edge_index)

        # Get source and destination node embeddings for each edge
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_embed = node_embed[src_idx]
        dst_embed = node_embed[dst_idx]

        # Predict edge features
        edge_pred = self.head(edge_attr, src_embed, dst_embed)

        return edge_pred


def compute_edge_loss(pred, target, mask, loss_type: str = "l1"):
    """
    Compute loss for edge predictions.

    Only computes loss for PSW edges (where mask == 1).

    Args:
        pred: Predicted ratios [num_edges, 2]
        target: Ground truth ratios [num_edges, 2]
        mask: Edge mask [num_edges] (1 for PSW, 0 for others)
        loss_type: 'l1' (paper default) or 'mse'

    Returns:
        Scalar loss value
    """
    # Expand mask to match prediction dimensions
    mask = mask.unsqueeze(-1).expand_as(pred)

    # Masked predictions and targets
    masked_pred = pred * mask
    masked_target = target * mask

    # Count valid edges
    num_valid = mask.sum() + 1e-8

    if loss_type == "l1":
        loss = F.l1_loss(masked_pred, masked_target, reduction="sum") / num_valid
    else:
        loss = F.mse_loss(masked_pred, masked_target, reduction="sum") / num_valid

    return loss


def compute_metrics(pred, target, mask, threshold: float = 0.1):
    """
    Compute evaluation metrics for edge predictions.

    Args:
        pred: Predicted ratios [num_edges, 2]
        target: Ground truth ratios [num_edges, 2]
        mask: Edge mask [num_edges]
        threshold: Threshold for considering a wall exists

    Returns:
        Dictionary of metrics
    """
    mask_bool = mask.bool()

    if mask_bool.sum() == 0:
        return {"mae": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    pred_masked = pred[mask_bool]
    target_masked = target[mask_bool]

    # MAE
    mae = F.l1_loss(pred_masked, target_masked).item()

    # Binary classification metrics
    pred_binary = (pred_masked > threshold).float()
    target_binary = (target_masked > threshold).float()

    # Accuracy
    correct = (pred_binary == target_binary).float().mean().item()

    # Precision & Recall
    tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()

    return {
        "mae": mae,
        "accuracy": correct,
        "precision": precision,
        "recall": recall,
    }
