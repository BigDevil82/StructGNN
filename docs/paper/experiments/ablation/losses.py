"""
消融实验的损失函数模块

根据配置动态组合不同的损失函数
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from docs.paper.experiments.ablation.config import AblationConfig


class AblationHybridLoss(nn.Module):
    """
    可配置的混合损失函数

    根据AblationConfig中的开关，动态组合不同的损失项
    """

    def __init__(self, config: AblationConfig, target_densities: list = None):
        super().__init__()
        self.config = config

        # 基础损失
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss(reduction="none")

        # 目标密度（用于GlobalDensityLoss）
        if target_densities is None:
            target_densities = [1.57, 3.81, 8.71]  # 默认值
        self.register_buffer(
            "target_densities",
            torch.tensor(target_densities, dtype=torch.float),
        )

    def forward(
        self,
        pred_prob: torch.Tensor,
        pred_ratio: torch.Tensor,
        batch: Data,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算总损失

        Returns:
            total_loss: 加权总损失
            loss_dict: 各项损失的字典（用于日志记录）
        """
        target_ratio = batch.y
        constraint_mask = batch.constraint_mask
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_prob.device)

        # === 1. BCE分类损失 ===
        if self.config.use_bce_loss:
            target_cls = (target_ratio > 0.01).float()
            bce_loss = self.bce(pred_prob, target_cls)
            loss_dict["bce"] = bce_loss.item()
            total_loss = total_loss + self.config.cls_weight * bce_loss

        # === 2. MSE回归损失 ===
        if self.config.use_mse_loss:
            target_cls = (target_ratio > 0.01).float()
            reg_loss_all = self.mse(pred_ratio, target_ratio)
            valid_mask = target_cls * constraint_mask
            num_pos = valid_mask.sum() + 1e-6
            mse_loss = (reg_loss_all * valid_mask).sum() / num_pos
            loss_dict["mse"] = mse_loss.item()
            total_loss = total_loss + self.config.reg_weight * mse_loss

        # === 3. Vector IoU损失 ===
        if self.config.use_iou_loss:
            iou_loss = self._compute_iou_loss(pred_ratio, target_ratio)
            loss_dict["iou"] = iou_loss.item()
            total_loss = total_loss + self.config.iou_weight * iou_loss

        # === 4. 一致性损失 ===
        if self.config.use_consistency_loss:
            consistency_loss = self._compute_consistency_loss(pred_ratio, edge_index, edge_attr)
            loss_dict["consistency"] = consistency_loss.item()
            total_loss = total_loss + self.config.consistency_weight * consistency_loss

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def compute_density_loss(
        self,
        pred_ratio: torch.Tensor,
        batch_idx: torch.Tensor,
        condition: torch.Tensor,
        constraint_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算全局密度损失"""
        if not self.config.use_density_loss:
            return torch.tensor(0.0, device=pred_ratio.device)

        effective_pred = pred_ratio * constraint_mask
        node_density = effective_pred.sum(dim=1, keepdim=True)
        graph_density_pred = global_mean_pool(node_density, batch_idx)
        graph_density_target = torch.matmul(condition, self.target_densities.to(condition.device))

        return nn.functional.mse_loss(graph_density_pred.squeeze(-1), graph_density_target)

    def _compute_iou_loss(
        self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
    ) -> torch.Tensor:
        """Vector IoU Loss"""
        intersection = torch.min(pred, target)
        union = torch.max(pred, target)
        i_sum = torch.sum(intersection, dim=1)
        u_sum = torch.sum(union, dim=1)
        iou = (i_sum + smooth) / (u_sum + smooth)
        return (1.0 - iou).mean()

    def _compute_consistency_loss(
        self,
        pred_ratio: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """相邻房间一致性损失"""
        src_idx, dst_idx = edge_index
        pred_src = pred_ratio[src_idx]
        pred_dst = pred_ratio[dst_idx]

        is_top = edge_attr[:, 3] == 1
        is_bottom = edge_attr[:, 4] == 1
        is_left = edge_attr[:, 5] == 1
        is_right = edge_attr[:, 6] == 1

        loss = torch.tensor(0.0, device=pred_ratio.device)
        mse = nn.MSELoss()

        # Right-Left对应
        if is_right.any():
            loss = loss + mse(pred_src[is_right, 4:8], pred_dst[is_right, 12:16])

        # Left-Right对应
        if is_left.any():
            loss = loss + mse(pred_src[is_left, 12:16], pred_dst[is_left, 4:8])

        # Top-Bottom对应
        if is_top.any():
            loss = loss + mse(pred_src[is_top, 0:4], pred_dst[is_top, 8:12])

        # Bottom-Top对应
        if is_bottom.any():
            loss = loss + mse(pred_src[is_bottom, 8:12], pred_dst[is_bottom, 0:4])

        return loss
