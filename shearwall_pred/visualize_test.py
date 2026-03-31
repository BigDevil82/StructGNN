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
from shearwall_pred.config import training_config, viz_config
from shearwall_pred.model import ShearWallGNN
from shearwall_pred.utils import build_graph_from_dxf, mask_to_constraint_vector


def prepare_graph_data_for_inference(dxf_path: str, mode: str = "none", category: Optional[int] = None):
    """
    现在的逻辑非常简单：获取 builder -> 转 batch
    """
    from torch_geometric.data import Batch

    # 获取 Builder
    builder = build_graph_from_dxf(dxf_path, mode)

    # 转 PyG 数据
    data = builder.to_pyg_data()
    # 强制修改 category（如果提供的话）
    if category is not None:
        cate_one_hot = np.zeros((1, 3))
        cate_one_hot[:, category] = 1.0
        data.condition = torch.tensor(cate_one_hot, dtype=torch.float)
    data_batch = Batch.from_data_list([data]).to(training_config.DEVICE)

    # 直接返回 builder，因为它包含了 rooms 和 graph 结构，用于后续画图
    return data_batch, builder


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


def visualize_single_case(
    dxf_path: str,
    model: ShearWallGNN,
    save_path: Optional[str] = None,
    mode: str = "none",
    category: Optional[int] = None,
) -> float:
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
    data_batch, builder = prepare_graph_data_for_inference(dxf_path, mode=mode, category=category)

    # 2. 模型预测
    predictions = predict_shear_walls(model, data_batch)

    # 3. 可视化绘图
    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=viz_config.FIG_SIZE_DOUBLE)

    # 上图：Ground Truth
    # ax_gt.set_title("Ground Truth (真实分布)", fontsize=14, fontweight="bold")
    for node_id in builder.graph.nodes:
        # 从节点属性中直接获取数据
        node_data = builder.graph.nodes[node_id]

        room_poly = node_data["poly"]  # 之前叫 calibrated_rooms[i]
        gt_vec = node_data.get("sw_vector")  # 之前要在 list 里查
        masks = node_data.get("masks", [])

        if gt_vec is not None:
            plot_room_analysis(room_poly, gt_vec, masks, ax_gt, wall_color=viz_config.GT_WALL_COLOR)

    # 下图：Prediction
    # ax_pred.set_title("Model Prediction (模型预测)", fontsize=14, fontweight="bold")
    iou_scores = []

    node_ids = list(builder.graph.nodes())

    for i, node_id in enumerate(node_ids):
        node_data = builder.graph.nodes[node_id]
        room_poly = node_data["poly"]
        masks = node_data.get("masks", [])
        gt_vec = node_data.get("sw_vector")
        pred_vec = predictions[i]  # 对应第 i 个节点

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

        # # 在房间中心显示IoU
        # c = room_poly.centroid
        # ax_pred.text(c.x, c.y, f"{iou:.2f}", color="black", fontsize=6, fontweight="bold")

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


def test_conditional_predict(
    dxf_path: str,
    model: ShearWallGNN,
    save_path: Optional[str] = None,
):
    """
    测试同一输入下3种类别条件的预测效果
    """

    ious = []
    for i in range(3):
        # 构建保存路径
        if save_path:
            base_path = Path(save_path)
            save_path_i = base_path.parent / f"{base_path.stem}_cate{i}{base_path.suffix}"
        else:
            save_path_i = None

        iou = visualize_single_case(dxf_path, model, save_path=save_path_i, category=i)
        ious.append(iou)
    return ious


def conditional_pred_on_testset():
    from shearwall_pred.config import data_config, model_config, training_config
    from shearwall_pred.cross_validate import EnsembleShearWallGNN

    cv_path = Path(data_config.SAVE_DIR)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.to(training_config.DEVICE)

    # dxf = r"data\data\dxf\dataset_split_8_2\test\L17_115.dxf"
    cate_ious = {0: [], 1: [], 2: []}
    for fname in os.listdir(data_config.DXF_DIR + "/test"):
        if fname.endswith(".dxf"):
            dxf = os.path.join(data_config.DXF_DIR, "test", fname)
            save_dir = str(cv_path / "conditional_test_results")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, os.path.basename(dxf).replace(".dxf", ".png"))
            ious = test_conditional_predict(dxf_path=dxf, model=ensemble_model, save_path=save_path)

            for i, iou in enumerate(ious):
                cate_ious[i].append(iou)
