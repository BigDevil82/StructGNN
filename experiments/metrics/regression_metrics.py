"""
回归指标计算模块

提供墙体长度比例预测的回归性能评估：MAE, RMSE
"""

import numpy as np


def compute_regression_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask_threshold: float = 0.01,
) -> dict:
    """
    计算回归指标（仅在有墙位置）

    仅在Ground Truth有墙的位置计算误差，避免零值膨胀对指标的影响。

    Args:
        pred: 预测值，形状 (N, 16)，其中N是节点数量
        gt: Ground Truth，形状 (N, 16)
        mask_threshold: 掩码阈值，GT中大于此值的位置被认为有墙

    Returns:
        dict: 包含以下指标
            - mae: 平均绝对误差 (Mean Absolute Error)
            - rmse: 均方根误差 (Root Mean Squared Error)
            - mae_per_sample: 每个样本的MAE，形状 (N,)
            - valid_ratio: 有效位置比例（用于参考）

    示例:
        >>> pred = np.array([[0.8, 0.05, ...], ...])  # (N, 16)
        >>> gt = np.array([[0.75, 0.0, ...], ...])
        >>> metrics = compute_regression_metrics(pred, gt)
        >>> print(f"MAE: {metrics['mae']:.4f}")
    """
    # 1. 创建掩码：仅在GT有墙的位置计算
    mask = gt > mask_threshold  # (N, 16)

    # 2. 过滤有效位置
    valid_pred = pred[mask]
    valid_gt = gt[mask]

    # 3. 计算全局指标
    if len(valid_pred) == 0:
        # 边界情况：没有有效位置
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mae_per_sample": np.zeros(pred.shape[0]),
            "valid_ratio": 0.0,
        }

    # Mean Absolute Error
    mae = float(np.mean(np.abs(valid_pred - valid_gt)))

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((valid_pred - valid_gt) ** 2)))

    # 4. 每个样本的MAE（用于分析分布）
    # 计算每个样本的误差，然后除以该样本的有效位置数
    abs_error = np.abs(pred - gt) * mask  # 仅保留有效位置的误差
    valid_count_per_sample = mask.sum(axis=1)  # 每个样本的有效位置数

    # 避免除零
    mae_per_sample = np.sum(abs_error, axis=1) / (valid_count_per_sample + 1e-6)

    # 5. 有效位置比例（用于参考）
    valid_ratio = float(mask.sum() / mask.size)

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_per_sample": mae_per_sample,
        "valid_ratio": valid_ratio,
    }


def compute_per_edge_regression_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask_threshold: float = 0.01,
) -> dict:
    """
    计算每条边（4条边）的回归指标

    用于分析模型在不同边上的回归性能差异

    Args:
        pred: 形状 (N, 16)
        gt: 形状 (N, 16)
        mask_threshold: 掩码阈值

    Returns:
        dict: 包含 edge_0, edge_1, edge_2, edge_3 的指标
            每个edge包含 {mae, rmse}
    """
    edge_names = ["top", "bottom", "left", "right"]
    results = {}

    for edge_idx in range(4):
        # 每条边有4个端点
        start_idx = edge_idx * 4
        end_idx = start_idx + 4

        edge_metrics = compute_regression_metrics(
            pred[:, start_idx:end_idx],
            gt[:, start_idx:end_idx],
            mask_threshold,
        )

        results[edge_names[edge_idx]] = {
            "mae": edge_metrics["mae"],
            "rmse": edge_metrics["rmse"],
        }

    return results


def compute_regression_metrics_by_length_range(
    pred: np.ndarray,
    gt: np.ndarray,
    ranges: list = None,
) -> dict:
    """
    按墙体长度范围分组计算回归指标

    Args:
        pred: 形状 (N, 16)
        gt: 形状 (N, 16)
        ranges: 长度范围列表，如 [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]

    Returns:
        dict: 每个范围的MAE和RMSE
    """
    if ranges is None:
        ranges = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]

    results = {}

    for range_min, range_max in ranges:
        # 筛选在此范围内的GT
        mask = (gt >= range_min) & (gt < range_max)

        if mask.sum() == 0:
            continue

        valid_pred = pred[mask]
        valid_gt = gt[mask]

        mae = float(np.mean(np.abs(valid_pred - valid_gt)))
        rmse = float(np.sqrt(np.mean((valid_pred - valid_gt) ** 2)))

        range_name = f"{range_min:.1f}-{range_max:.1f}"
        results[range_name] = {
            "mae": mae,
            "rmse": rmse,
            "count": int(mask.sum()),
        }

    return results
