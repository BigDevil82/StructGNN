"""
损失函数模块
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from shearwall_pred.config import training_config


class VectorIoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: (Batch, 16) - 模型的回归输出 (经过 Sigmoid)
        target: (Batch, 16) - 真实标签
        """
        # 1. 确保 pred 在 [0,1] (通常已经在外面做过 sigmoid，这里为了保险)
        # 注意：不要在 Loss 里再次做 sigmoid，除非外部没做

        # 2. 计算交集和并集
        # PyTorch 的 min/max 可以通过反向传播梯度
        intersection = torch.min(pred, target)
        union = torch.max(pred, target)

        # 3. 按样本求和
        i_sum = torch.sum(intersection, dim=1)
        u_sum = torch.sum(union, dim=1)

        # 4. 计算 IoU
        iou = (i_sum + self.smooth) / (u_sum + self.smooth)

        # 5. Loss = 1 - IoU
        loss = 1.0 - iou

        return loss.mean()


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
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

        return loss


class GlobalDensityLoss(nn.Module):
    def __init__(self):
        """
        全局密度约束损失
        Args:
            target_densities: 列表 [d0, d1, d2]，对应3种设计条件的预期密度目标。
                              数值代表：平均每个节点（房间）所有墙预测值的总和。
                              例如: [1.5, 3.0, 5.0] 表示低/中/高密度
                              [3.569831529585253, 4.812471681428187, 6.708384769004688]
        """
        super().__init__()
        target_densities = [1.569831529585253, 3.812471681428187, 8.708384769004688]
        self.register_buffer(
            "target_densities", torch.tensor(target_densities, device=training_config.DEVICE)
        )

        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_ratio: torch.Tensor,
        batch_idx: torch.Tensor,
        condition: torch.Tensor,
        constraint_mask: torch.Tensor,
    ):
        """
        Args:
            pred_ratio: (N, 16) 节点的回归预测输出
            batch_idx: (N,) 节点属于哪个图的索引 (data.batch)
            condition: (Batch_Size, 3) 图级的设计条件 One-hot 或软标签
        """
        # 1. 计算每个节点的“墙体总量” (N, )
        # 将16个方位的预测值相加，代表这个房间一共布置了多少墙
        effective_pred = pred_ratio * constraint_mask
        node_density = effective_pred.sum(dim=1, keepdim=True)  # (N, 1)

        # 2. 聚合到图级别 (Batch_Size, )
        # 计算整张图所有房间的平均墙体总量
        graph_density_pred = global_mean_pool(node_density, batch_idx)

        # 3. 计算目标密度 (Batch_Size, )
        # condition (B, 3) * targets (3,) -> (B,)
        # 这一步根据每张图的条件，取出对应的目标密度值
        graph_density_target = torch.matmul(condition, self.target_densities)

        # 4. 计算损失
        # 强迫预测的平均密度 接近 目标密度
        return self.mse(graph_density_pred.squeeze(-1), graph_density_target)


class HybridLoss(nn.Module):
    """
    混合损失函数（分类 + 回归）

    将问题建模为两阶段：
    1. 分类：该位置是否有剪力墙 (BCE Loss)
    2. 回归：剪力墙的长度比例 (MSE Loss，仅在有墙的位置计算)
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 2.0, consistency_weight: float = 0.5):
        """
        Args:
            cls_weight: 分类损失权重
            reg_weight: 回归损失权重
        """
        super().__init__()
        self.bce = nn.BCELoss()  # 二分类交叉熵
        self.mse = nn.MSELoss(reduction="none")  # 回归损失（不约减）
        self.iou_loss = VectorIoULoss()
        self.consistency_loss = ConsistencyLoss()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.consis_weight = consistency_weight

    def forward(
        self,
        pred_prob: torch.Tensor,
        pred_ratio: torch.Tensor,
        batch: Data,
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
        target_ratio, constraint_mask, edge_index, edge_attr = (
            batch.y,
            batch.constraint_mask,
            batch.edge_index,
            batch.edge_attr,
        )

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

        loss_iou = self.iou_loss(pred_ratio, target_ratio)

        loss_consistency = self.consistency_loss(pred_ratio, edge_index, edge_attr)

        # 4. 加权组合
        return (
            self.cls_weight * cls_loss
            + self.reg_weight * reg_loss
            + 2 * loss_iou
            + self.consis_weight * loss_consistency
        )
