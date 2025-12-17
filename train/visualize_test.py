"""
测试集可视化模块

功能：
1. 加载DXF文件并进行预测
2. 对比Ground Truth和模型预测结果
3. 计算IoU指标
4. 生成可视化对比图
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon

from preprocess.room_analyzer import calculate_wall_iou, plot_room_analysis, reconstruct_walls
from train.config import viz_config
from train.model import ShearWallGNN
from train.utils import build_graph_from_dxf, mask_to_constraint_vector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_graph_data_for_inference(dxf_path: str):
    """
    从DXF文件准备用于推理的图数据

    Args:
        dxf_path: DXF文件路径

    Returns:
        (data_batch, calibrated_rooms, analysis_results, node_ids):
            - data_batch: PyG Batch对象（单个图）
            - calibrated_rooms: 校准后的房间列表
            - analysis_results: Ground Truth分析结果
            - node_ids: 节点ID列表（用于映射回房间）
    """
    from torch_geometric.data import Batch, Data

    # 构建图
    graph_builder, calibrated_rooms, analysis_results = build_graph_from_dxf(dxf_path)
    G = graph_builder.graph

    # 准备特征
    x_list = []
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    node_ids = list(G.nodes())

    for node in node_ids:
        node_data = G.nodes[node]
        geo_feature = node_data["geo_feature"]
        masks = node_data.get("masks", [])
        constraint_vector = mask_to_constraint_vector(masks)

        # 拼接输入特征
        x_feat = np.concatenate([geo_feature, constraint_vector])
        x_list.append(x_feat)

    # 构建边
    edge_index = []
    edge_attr = []
    for u, v, edge_data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_attr.append(edge_data["feature"])

    # 转换为Tensor
    x = torch.tensor(np.array(x_list), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    # 创建PyG Data并转为Batch
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_batch = Batch.from_data_list([data]).to(DEVICE)

    return data_batch, calibrated_rooms, analysis_results, node_ids


def predict_shear_walls(model: ShearWallGNN, data_batch) -> np.ndarray:
    """
    使用模型预测剪力墙分布

    Args:
        model: 训练好的模型
        data_batch: PyG Batch对象

    Returns:
        predictions: 预测结果 (N_nodes, 16)，经过阈值处理
    """
    model.eval()
    with torch.no_grad():
        pred_prob, pred_ratio = model(data_batch)

        # 组合分类和回归结果：只有概率>阈值的位置才保留回归值
        pred_combined = (pred_prob > viz_config.PRED_PROB_THRESHOLD) * pred_ratio
        predictions = pred_combined.cpu().numpy()

        # 过滤小值
        predictions = np.where(predictions < viz_config.PRED_RATIO_THRESHOLD, 0.0, predictions)

    return predictions


def visualize_single_case(dxf_path: str, model: ShearWallGNN, save_path: Optional[str] = None) -> float:
    """
    可视化单个案例的预测结果

    流程：
    1. 提取DXF → 构建图 → 模型预测
    2. 重建Ground Truth和预测墙体
    3. 计算IoU指标
    4. 并排绘制对比图

    Args:
        dxf_path: DXF文件路径
        model: 训练好的模型
        save_path: 保存路径（可选）

    Returns:
        avg_iou: 平均IoU分数
    """
    print(f"正在处理: {os.path.basename(dxf_path)}")

    # 1. 准备数据
    data_batch, calibrated_rooms, analysis_results, node_ids = prepare_graph_data_for_inference(dxf_path)

    # 2. 模型预测
    predictions = predict_shear_walls(model, data_batch)

    # 3. 可视化绘图
    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=viz_config.FIG_SIZE_DOUBLE)

    # 左图：Ground Truth
    ax_gt.set_title("Ground Truth (真实分布)", fontsize=14, fontweight="bold")
    for res in analysis_results:
        idx = res["room_index"]
        if idx < len(calibrated_rooms):
            room_poly = calibrated_rooms[idx]
            gt_vec = np.array(res["sw_vector"])
            masks = res["masks"]

            plot_room_analysis(room_poly, gt_vec, masks, ax_gt, wall_color=viz_config.GT_WALL_COLOR)

            # 标注房间号
            c = room_poly.centroid
            ax_gt.text(c.x, c.y, str(idx), color="blue", fontsize=10)

    # 右图：Prediction
    ax_pred.set_title("Model Prediction (模型预测)", fontsize=14, fontweight="bold")
    iou_scores = []

    # 遍历每个节点，绘制预测结果
    for i, node_id in enumerate(node_ids):
        if node_id >= len(calibrated_rooms):
            continue

        room_poly = calibrated_rooms[node_id]
        pred_vec = predictions[i]

        # 获取对应的Ground Truth和Masks
        gt_vec = np.zeros(16)
        masks = []
        for res in analysis_results:
            if res["room_index"] == node_id:
                gt_vec = np.array(res["sw_vector"])
                masks = res["masks"]
                break

        # 重建墙体
        pred_walls = reconstruct_walls(room_poly, pred_vec, masks)
        gt_walls = reconstruct_walls(room_poly, gt_vec, masks)

        # 绘图
        plot_room_analysis(
            room_poly, pred_vec, masks, ax_pred, walls=pred_walls, wall_color=viz_config.PRED_WALL_COLOR
        )

        # 计算并显示IoU
        iou = calculate_wall_iou(gt_walls, pred_walls, buffer_width=viz_config.IOU_BUFFER_WIDTH)
        iou_scores.append(iou)

        # 在房间中心显示IoU
        c = room_poly.centroid
        ax_pred.text(c.x, c.y, f"{iou:.2f}", color="black", fontsize=9, fontweight="bold")

    # 设置样式
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    fig.suptitle(f"File: {os.path.basename(dxf_path)} | Avg IoU: {avg_iou:.4f}", fontsize=16)

    for ax in [ax_gt, ax_pred]:
        ax.set_aspect("equal")
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=viz_config.DPI, bbox_inches="tight")
        print(f"结果已保存至: {save_path}")
    else:
        plt.show()

    plt.close()
    return avg_iou
