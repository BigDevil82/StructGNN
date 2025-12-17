import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon

from dxf_extractor import DXFExtractor
from preprocess.layout_graph import LayoutGraphBuilder
from preprocess.room_analyzer import RoomAnalyzer, calculate_wall_iou, plot_room_analysis, reconstruct_walls
from preprocess.room_calibrator import calibrate_rooms
from train.dataset import compute_anchor_ratios
from train.model import ShearWallGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_to_vector(buildable_masks_list):
    """复用 dataset.py 中的逻辑，确保特征一致"""
    vector = np.ones(16, dtype=np.float32)
    buildable_ratios = []
    for intervals in buildable_masks_list:
        start_ratio, end_ratio = compute_anchor_ratios(intervals)
        buildable_ratios.extend([start_ratio, end_ratio])
    for i, ratio in enumerate(buildable_ratios):
        if ratio < 0.4:
            vector[i] = 0.0
    return vector


def visualize_single_case(dxf_path, model: ShearWallGNN, save_path=None):
    """
    流程：提取DXF -> 构建图 -> 模型预测 -> 可视化对比
    """
    print(f"正在处理: {os.path.basename(dxf_path)}")

    # 1. 提取与校准 (为了获取几何信息用于绘图)
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

    if not raw_rooms:
        print("无有效房间")
        return

    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)

    # 2. GT 分析 (获取 Ground Truth 和 Masks)
    analyzer = RoomAnalyzer(sw_polys, infill_polys)
    analysis_results = []
    # 注意：这里逻辑要和 dataset.py 保持严格一致
    for i, room in enumerate(raw_rooms):
        if room.is_valid and room.area > 1:
            sw, ms = analyzer.process_room(room)
            analysis_results.append({"room_index": i, "sw_vector": sw, "masks": ms})

    # 3. 构建图 (为了获取特征喂给模型)
    gb = LayoutGraphBuilder(calibrated_rooms)
    gb.add_analysis_results(analysis_results)
    G = gb.graph

    # 4. 转换为 Tensor (模拟 DataLoader 的行为)
    from torch_geometric.data import Batch, Data

    x_list = []
    mask_feat_list = []
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    node_ids = list(G.nodes())  # 保持顺序以便后续映射回房间

    for node in node_ids:
        node_data = G.nodes[node]
        geo = node_data["geo_feature"]

        # 处理 Masks 特征
        constraint_vec = mask_to_vector(node_data["masks"])

        # 拼接输入: [Geo(9) + Constraint(16)]
        x_feat = np.concatenate([geo, constraint_vec])

        x_list.append(x_feat)
        mask_feat_list.append(constraint_vec)

    edge_index = []
    edge_attr = []
    for u, v, edge_data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_attr.append(edge_data["feature"])

    # 转换为 PyG Batch (虽然只有一个图，但模型通常预期有batch维)
    x = torch.tensor(np.array(x_list), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = Batch.from_data_list([data]).to(DEVICE)  # 增加 batch 维度

    # 5. 模型推理
    model.eval()
    with torch.no_grad():
        pred_prob, pred_ratio = model(data)  # Output: (N_nodes, 16)
        pred_out = (pred_prob > 0.5) * pred_ratio  # 逐元素相乘得到最终预测
        pred_out = pred_out.cpu().numpy()

    # 6. 可视化绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # 左图：Ground Truth
    ax1.set_title("Ground Truth (真实分布)")
    for res in analysis_results:
        idx = res["room_index"]
        # 只有在 calibrated_rooms 对应的房间才绘制
        if idx < len(calibrated_rooms):
            room_poly = calibrated_rooms[idx]
            gt_vec = np.array(res["sw_vector"])
            masks = res["masks"]
            plot_room_analysis(room_poly, gt_vec, masks, ax1, wall_color="green")

            # 标注房间号
            c = room_poly.centroid
            ax1.text(c.x, c.y, str(idx), color="blue", fontsize=10)

    # 右图：Prediction
    ax2.set_title("Model Prediction (模型预测)")
    iou_scores = []

    # 遍历每个节点，找到对应的房间并绘制预测结果
    for i, node_id in enumerate(node_ids):
        # node_id 对应的是 analysis_results 中的 room_index (如果 LayoutGraphBuilder 逻辑没变)
        if node_id < len(calibrated_rooms):
            room_poly = calibrated_rooms[node_id]

            # 获取该房间的预测向量
            pred_vec = pred_out[i]
            # 将小于0.2的值置0，模拟阈值处理
            pred_vec = np.where(pred_vec < 0.1, 0.0, pred_vec)

            # 获取对应的 Masks (用于后处理裁剪)
            # 注意：需在 analysis_results 中找到对应数据
            masks = []
            gt_vec = np.zeros(16)
            for res in analysis_results:
                if res["room_index"] == node_id:
                    masks = res["masks"]
                    gt_vec = np.array(res["sw_vector"])
                    break

            # 重建墙体 (核心步骤)
            pred_walls = reconstruct_walls(room_poly, pred_vec, masks)
            gt_walls = reconstruct_walls(room_poly, gt_vec, masks)  # 重新生成GT墙体用于计算IoU

            # 绘图
            plot_room_analysis(room_poly, pred_vec, masks, ax2, walls=pred_walls, wall_color="red")

            # 计算并显示 IoU
            iou = calculate_wall_iou(gt_walls, pred_walls)
            iou_scores.append(iou)

            # 在房间中心显示 IoU
            c = room_poly.centroid
            ax2.text(c.x, c.y, f"{iou:.2f}", color="black", fontsize=9, fontweight="bold")

    # 设置样式
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    fig.suptitle(f"File: {os.path.basename(dxf_path)} | Avg IoU: {avg_iou:.4f}", fontsize=16)

    for ax in [ax1, ax2]:
        ax.set_aspect("equal")
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"结果已保存至: {save_path}")
    else:
        plt.show()
    plt.close()
    return avg_iou
