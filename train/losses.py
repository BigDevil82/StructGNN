"""
损失函数模块

提供两种损失函数：
1. PhysicsInformedLoss: 物理约束损失（基于掩码的惩罚项）
2. HybridLoss: 混合损失（分类 + 回归）
"""

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    物理约束损失函数

    在基础MSE损失基础上，增加物理约束惩罚项：
    - 如果在不可布置区域（mask=0）预测了墙体，则施加惩罚
    """

    def __init__(self, mask_penalty_weight: float = 2.0):
        """
        Args:
            mask_penalty_weight: 物理约束惩罚权重
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.penalty_weight = mask_penalty_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, constraint_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 模型预测 (N, 16)
            target: 真实标签 (N, 16)
            constraint_mask: 可布置区域掩码 (N, 16)
                           1=可布置区域, 0=不可布置区域（门窗等）

        Returns:
            total_loss: 总损失 = MSE损失 + 物理约束惩罚
        """
        # 1. 基础回归损失
        basic_loss = self.mse(pred, target)

        # 2. 物理约束惩罚
        # 在不可布置区域（mask=0）的预测值应该为0
        # 惩罚项 = pred * (1 - mask)
        # - 如果mask=1（可布置），则(1-mask)=0，无惩罚
        # - 如果mask=0（不可布置），则(1-mask)=1，惩罚预测值
        violation = pred * (1 - constraint_mask)
        penalty_loss = torch.mean(violation**2)

        return basic_loss + self.penalty_weight * penalty_loss


class HybridLoss(nn.Module):
    """
    混合损失函数（分类 + 回归）

    将问题建模为两阶段：
    1. 分类：该位置是否有剪力墙 (BCE Loss)
    2. 回归：剪力墙的长度比例 (MSE Loss，仅在有墙的位置计算)
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 2.0):
        """
        Args:
            cls_weight: 分类损失权重
            reg_weight: 回归损失权重
        """
        super().__init__()
        self.bce = nn.BCELoss()  # 二分类交叉熵
        self.mse = nn.MSELoss(reduction="none")  # 回归损失（不约减）
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        pred_prob: torch.Tensor,
        pred_ratio: torch.Tensor,
        target_ratio: torch.Tensor,
        constraint_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_prob: 分类预测概率 (N, 16)
            pred_ratio: 回归预测比例 (N, 16)
            target_ratio: 真实比例 (N, 16)
            constraint_mask: 可布置区域掩码 (N, 16)

        Returns:
            total_loss: 加权总损失
        """
        # 1. 生成分类标签
        # 如果真实长度 > 0.01，则认为该位置有墙
        target_cls = (target_ratio > 0.01).float()

        # 2. 分类损失 (BCE)
        cls_loss = self.bce(pred_prob, target_cls)

        # 3. 回归损失（只在有墙的位置计算）
        # 关键：如果GT没有墙，不应惩罚回归头的预测值，避免干扰学习
        reg_loss_all = self.mse(pred_ratio, target_ratio)

        # 构建有效掩码：
        # - pos_mask: Ground Truth中有墙的位置
        # - valid_mask: 既有墙，又在可布置区域内
        pos_mask = target_cls
        valid_mask = pos_mask * constraint_mask

        # 计算有效位置的平均回归损失
        num_pos = valid_mask.sum() + 1e-6  # 避免除以0
        reg_loss = (reg_loss_all * valid_mask).sum() / num_pos

        # 4. 加权组合
        return self.cls_weight * cls_loss + self.reg_weight * reg_loss
