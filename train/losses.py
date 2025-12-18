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


# 在 losses.py 中
class ConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()

    def forward(self, pred_ratio, edge_index, edge_attr):
        """
        pred_ratio: (N, 16) - 16维的回归输出
        edge_index: (2, E)
        edge_attr: (E, D) - 包含相对方位信息 [..., is_top, is_bottom, is_left, is_right, ...]
        """
        # 1. 获取每条边的源节点(src)和目标节点(dst)
        src_idx, dst_idx = edge_index

        # 2. 获取源节点和目标节点的预测值
        pred_src = pred_ratio[src_idx]  # (E, 16)
        pred_dst = pred_ratio[dst_idx]  # (E, 16)

        # 3. 确定对应关系 (Mapping)
        # 假设 16维向量顺序是:
        # [Top(4), Right(4), Bottom(4), Left(4)] (每个方向包含2段*2端点=4值)
        # 这里的索引需要根据你实际的 sw_vector 定义来写

        # 示例：如果 edge 是 "Right" (dst 在 src 的右边)
        # 那么 src 的 "Right Wall" 应该等于 dst 的 "Left Wall"

        # 假设 edge_attr 中第 3,4,5,6 位是 Top, Bottom, Left, Right one-hot
        is_top = edge_attr[:, 3] == 1
        is_bottom = edge_attr[:, 4] == 1
        is_left = edge_attr[:, 5] == 1
        is_right = edge_attr[:, 6] == 1
        loss = 0

        # === Case 1: Src 的右边是 Dst (即检查 Src_Right vs Dst_Left) ===
        # 假设 Right Wall 的索引是 4:8, Left Wall 的索引是 12:16
        if is_right.any():
            # 取出所有这种关系的边
            p_src = pred_src[is_right, 4:8]
            p_dst = pred_dst[is_right, 12:16]
            loss += self.mse(p_src, p_dst)

        # === Case 2: Src 的左边是 Dst (即检查 Src_Left vs Dst_Right) ===
        if is_left.any():
            p_src = pred_src[is_left, 12:16]
            p_dst = pred_dst[is_left, 4:8]
            loss += self.mse(p_src, p_dst)

        if is_top.any():
            p_src = pred_src[is_top, 0:4]
            p_dst = pred_dst[is_top, 8:12]
            loss += self.mse(p_src, p_dst)

        if is_bottom.any():
            p_src = pred_src[is_bottom, 8:12]
            p_dst = pred_dst[is_bottom, 0:4]
            loss += self.mse(p_src, p_dst)

        return self.weight * loss


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
