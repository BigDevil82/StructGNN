"""
分类指标计算模块

提供墙体存在性预测的分类性能评估：Precision, Recall, F1-Score
"""

import numpy as np


def compute_classification_metrics(
    pred_prob: np.ndarray,
    pred_ratio: np.ndarray,
    gt: np.ndarray,
    prob_threshold: float = 0.5,
    ratio_threshold: float = 0.01,
) -> dict:
    """
    计算分类指标（二分类：有墙/无墙）

    将每个端点位置视为一个二分类样本，评估模型对"是否有墙"的判断能力。

    Args:
        pred_prob: 分类概率，形状 (N, 16)，其中N是节点数量，16是每个节点的预测维度
        pred_ratio: 回归比例，形状 (N, 16)
        gt: Ground Truth，形状 (N, 16)
        prob_threshold: 概率阈值，高于此值认为预测有墙
        ratio_threshold: 比例阈值，高于此值认为GT有墙

    Returns:
        dict: 包含以下指标
            - precision: 精确率 (TP / (TP + FP))
            - recall: 召回率 (TP / (TP + FN))
            - f1: F1分数 (2 * P * R / (P + R))
            - accuracy: 准确率 ((TP + TN) / Total)
            - tp, fp, fn, tn: 混淆矩阵各项计数

    示例:
        >>> pred_prob = np.array([[0.9, 0.1, ...], ...])  # (N, 16)
        >>> pred_ratio = np.array([[0.8, 0.05, ...], ...])
        >>> gt = np.array([[0.75, 0.0, ...], ...])
        >>> metrics = compute_classification_metrics(pred_prob, pred_ratio, gt)
        >>> print(f"F1-Score: {metrics['f1']:.4f}")
    """
    # 1. 二值化GT：墙体比例大于阈值视为有墙
    gt_binary = (gt > ratio_threshold).astype(int).flatten()

    # 2. 二值化预测：结合分类概率和回归比例
    # 只有当分类概率高且回归比例高时，才认为预测有墙
    pred_combined = (pred_prob > prob_threshold) * pred_ratio
    pred_binary = (pred_combined > ratio_threshold).astype(int).flatten()

    # 3. 计算混淆矩阵
    tp = int(((pred_binary == 1) & (gt_binary == 1)).sum())  # True Positive
    fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())  # False Positive
    fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())  # False Negative
    tn = int(((pred_binary == 0) & (gt_binary == 0)).sum())  # True Negative

    # 4. 计算分类指标
    # 精确率：预测为正的样本中，真正为正的比例
    precision = tp / (tp + fp + 1e-6)

    # 召回率：实际为正的样本中，被正确预测为正的比例
    recall = tp / (tp + fn + 1e-6)

    # F1分数：精确率和召回率的调和平均
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # 准确率：所有样本中预测正确的比例
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-6)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_per_edge_classification_metrics(
    pred_prob: np.ndarray,
    pred_ratio: np.ndarray,
    gt: np.ndarray,
    prob_threshold: float = 0.5,
    ratio_threshold: float = 0.01,
) -> dict:
    """
    计算每条边（4条边）的分类指标

    用于分析模型在不同边上的表现差异（如：顶边、底边、左边、右边）

    Args:
        pred_prob: 形状 (N, 16)
        pred_ratio: 形状 (N, 16)
        gt: 形状 (N, 16)
        prob_threshold: 概率阈值
        ratio_threshold: 比例阈值

    Returns:
        dict: 包含 edge_0, edge_1, edge_2, edge_3 的指标
            每个edge包含 {precision, recall, f1}
    """
    edge_names = ["top", "bottom", "left", "right"]
    results = {}

    for edge_idx in range(4):
        # 每条边有4个端点（2个半边，每个半边2个端点）
        start_idx = edge_idx * 4
        end_idx = start_idx + 4

        edge_metrics = compute_classification_metrics(
            pred_prob[:, start_idx:end_idx],
            pred_ratio[:, start_idx:end_idx],
            gt[:, start_idx:end_idx],
            prob_threshold,
            ratio_threshold,
        )

        results[edge_names[edge_idx]] = {
            "precision": edge_metrics["precision"],
            "recall": edge_metrics["recall"],
            "f1": edge_metrics["f1"],
        }

    return results
