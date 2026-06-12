"""
模型对比可视化脚本

将Ours模型和Baseline模型的预测结果并排对比展示：
- 左列：Ground Truth
- 中列：Ours模型预测结果
- 右列：Baseline模型预测结果

每个样本保存为一张对比图
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon
from torch_geometric.data import Batch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from baseline model
from docs.paper.experiments.research.baseline_edge_gnn.config import data_config
from docs.paper.experiments.research.baseline_edge_gnn.config import model_config as baseline_data_config
from docs.paper.experiments.research.baseline_edge_gnn.ensemble_model import EnsembleGNNEP4
from docs.paper.experiments.research.baseline_edge_gnn.graph_builder import (
    build_graph_from_json,
    load_json_data,
)

# Import from ours model
from src.data_engine.preprocess.room_analyzer import plot_room_analysis, reconstruct_walls
from src.shearwall_pred.config import model_config, training_config, viz_config
from src.shearwall_pred.cross_validate import EnsembleShearWallGNN
from src.shearwall_pred.utils import build_graph_from_dxf


def load_ours_model(cv_dir: str, device: str = "cuda") -> EnsembleShearWallGNN:
    """加载Ours集成模型"""
    from pathlib import Path

    cv_path = Path(cv_dir)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))

    if not model_paths:
        raise FileNotFoundError(f"在 {cv_path} 下未找到任何 fold_*/best_model.pth")

    print(f"找到 {len(model_paths)} 个Ours模型检查点")
    model_paths_str = [str(p) for p in model_paths]

    ensemble_model = EnsembleShearWallGNN(model_paths_str, model_config)
    ensemble_model.to(device)
    print(f"Loaded Ours ensemble model from {cv_dir}")
    return ensemble_model


def load_baseline_model(cv_dir: str, device: str = "cuda") -> EnsembleGNNEP4:
    """加载Baseline集成模型"""
    device_obj = torch.device(device)
    ensemble_model = EnsembleGNNEP4.from_cv_dir(
        cv_dir=cv_dir,
        node_in_dim=baseline_data_config.NODE_IN_DIM,
        edge_in_dim=baseline_data_config.EDGE_IN_DIM,
        gnn_out_dim=baseline_data_config.GNN_OUT_DIM,
        dropout=baseline_data_config.DROPOUT,
        n_folds=None,
        device=device_obj,
    )
    print(f"Loaded Baseline ensemble model from {cv_dir}")
    return ensemble_model


def load_metrics(ours_json: str, baseline_json: str) -> Tuple[Dict, Dict]:
    """加载两个模型的指标"""
    with open(ours_json, "r", encoding="utf-8") as f:
        ours_metrics = {item["file_key"]: item for item in json.load(f)}

    with open(baseline_json, "r", encoding="utf-8") as f:
        baseline_metrics = {item["file_key"]: item for item in json.load(f)}

    return ours_metrics, baseline_metrics


def predict_ours(model: EnsembleShearWallGNN, dxf_path: str, device: str) -> Tuple:
    """使用Ours模型进行预测"""
    builder = build_graph_from_dxf(dxf_path, mode="none")
    data = builder.to_pyg_data()
    data_batch = Batch.from_data_list([data]).to(device)

    model.eval()
    with torch.no_grad():
        pred_prob, pred_ratio = model(data_batch)
        pred_combined = (pred_prob > viz_config.PRED_PROB_THRESHOLD) * pred_ratio
        predictions = pred_combined.cpu().numpy()
        predictions = np.where(predictions < viz_config.PRED_RATIO_THRESHOLD, 0.0, predictions)

    return builder, predictions


def predict_baseline(model: EnsembleGNNEP4, json_data: Dict, file_key: str, device: str) -> Tuple:
    """使用Baseline模型进行预测"""
    builder = build_graph_from_json(json_data, file_key, mode="none")
    if builder is None:
        return None, None

    data = builder.to_pyg_data()
    if data is None:
        return None, None

    batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        pred = model(batch)

    predictions = pred.cpu().numpy()
    predictions[predictions < 0.1] = 0.0

    return builder, predictions


def plot_baseline_result(ax, builder, predictions):
    """绘制Baseline模型结果（只绘制PSW边的填充墙和剪力墙）"""
    edge_idx = 0

    for edge in builder.edges:
        line = edge["line"]
        edge_type = edge["edge_type"]
        coords = list(line.coords)

        # Only process PSW edges
        if edge_type == data_config.EDGE_TYPE_PSW and edge_idx < len(predictions):
            ratio_start, ratio_end = predictions[edge_idx]

            # Draw full edge as fill wall (灰色填充墙)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color="#BBBBBB", linewidth=8, zorder=3)

            # Draw shear wall overlay (红色剪力墙)
            if ratio_start > 0.02:
                total_len = line.length
                sw_len = total_len * ratio_start
                sw_point = line.interpolate(sw_len)
                ax.plot(
                    [coords[0][0], sw_point.x],
                    [coords[0][1], sw_point.y],
                    color="red",
                    linewidth=4,
                    zorder=4,
                )

            if ratio_end > 0.02:
                total_len = line.length
                sw_len = total_len * ratio_end
                sw_point = line.interpolate(total_len - sw_len)
                ax.plot(
                    [coords[-1][0], sw_point.x],
                    [coords[-1][1], sw_point.y],
                    color="red",
                    linewidth=4,
                    zorder=4,
                )

        edge_idx += 2


def visualize_comparison(
    file_key: str,
    dxf_path: str,
    ours_model: EnsembleShearWallGNN,
    baseline_model: EnsembleGNNEP4,
    json_data: Dict,
    ours_metrics: Dict,
    baseline_metrics: Dict,
    save_path: str,
    device: str = "cuda",
):
    """
    对比可视化单个样本

    三列布局：
    - 左：Ground Truth + 文件名
    - 中：Ours预测 + IoU
    - 右：Baseline预测 + IoU
    """
    # 预测
    ours_builder, ours_pred = predict_ours(ours_model, dxf_path, device)
    baseline_builder, baseline_pred = predict_baseline(baseline_model, json_data, file_key, device)

    if baseline_builder is None or baseline_pred is None:
        print(f"Skipped {file_key}: baseline prediction failed")
        return

    # 创建3列子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左图：Ground Truth (使用ours_builder)
    ax_gt = axes[0]
    for node_id in ours_builder.graph.nodes:
        node_data = ours_builder.graph.nodes[node_id]
        room_poly = node_data["poly"]
        gt_vec = node_data.get("sw_vector")
        masks = node_data.get("masks", [])

        if gt_vec is not None:
            plot_room_analysis(room_poly, gt_vec, masks, ax_gt, wall_color=viz_config.GT_WALL_COLOR)

    ax_gt.set_aspect("equal")
    ax_gt.axis("off")
    ax_gt.set_title(f"reference", fontsize=14, fontweight="bold")

    # 中图：Ours预测
    ax_ours = axes[1]
    node_ids = list(ours_builder.graph.nodes())

    for i, node_id in enumerate(node_ids):
        node_data = ours_builder.graph.nodes[node_id]
        room_poly = node_data["poly"]
        masks = node_data.get("masks", [])
        pred_vec = ours_pred[i]

        pred_walls = reconstruct_walls(room_poly, pred_vec, masks)
        plot_room_analysis(
            room_poly, pred_vec, masks, ax_ours, walls=pred_walls, wall_color=viz_config.PRED_WALL_COLOR
        )

    ax_ours.set_aspect("equal")
    ax_ours.axis("off")

    # 获取Ours的IoU
    ours_iou = ours_metrics.get(file_key, {}).get("image_iou", 0.0)
    ax_ours.set_title(f"Ours (IoU: {ours_iou:.4f})", fontsize=14, fontweight="bold")

    # 右图：Baseline预测
    ax_baseline = axes[2]
    plot_baseline_result(ax_baseline, baseline_builder, baseline_pred)

    ax_baseline.set_aspect("equal")
    ax_baseline.axis("off")
    ax_baseline.invert_yaxis()  # 与baseline保持一致

    # 获取Baseline的IoU
    baseline_iou = baseline_metrics.get(file_key, {}).get("image_iou", 0.0)
    ax_baseline.set_title(f"Baseline (IoU: {baseline_iou:.4f})", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Ours vs Baseline model predictions")
    parser.add_argument(
        "--ours_cv_dir",
        type=str,
        default="outputs/result/shearwall_pred/0126_cond_kfold",
        help="Path to Ours CV models directory",
    )
    parser.add_argument(
        "--baseline_cv_dir",
        type=str,
        default="outputs/result/shearwall_pred/baseline-GNN-EP4-CV",
        help="Path to Baseline CV models directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/dxf/shearwall_split_8_2/test",
        help="Test DXF files directory",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=data_config.JSON_PATH,
        help="JSON data file for baseline",
    )
    parser.add_argument(
        "--ours_metrics",
        type=str,
        default="outputs/result/metrics/image_iou_comparison/ours_per_sample.json",
        help="Ours model metrics JSON",
    )
    parser.add_argument(
        "--baseline_metrics",
        type=str,
        default="outputs/result/metrics/image_iou_comparison/baseline_per_sample.json",
        help="Baseline model metrics JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/result/shearwall_pred/comparison_plots",
        help="Output directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to visualize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("Loading models...")
    ours_model = load_ours_model(args.ours_cv_dir, args.device)
    baseline_model = load_baseline_model(args.baseline_cv_dir, args.device)

    # 加载指标
    print("\nLoading metrics...")
    ours_metrics, baseline_metrics = load_metrics(args.ours_metrics, args.baseline_metrics)

    # 加载JSON数据（用于baseline）
    print("\nLoading JSON data...")
    json_data = load_json_data(args.json_path)

    # 获取测试集文件列表
    test_files = []
    for fname in os.listdir(args.test_dir):
        if fname.lower().endswith(".dxf"):
            file_key = os.path.splitext(fname)[0]
            test_files.append((file_key, os.path.join(args.test_dir, fname)))

    if args.max_samples:
        test_files = test_files[: args.max_samples]

    print(f"\nProcessing {len(test_files)} test samples...")
    print("-" * 60)

    # 处理每个测试样本
    for file_key, dxf_path in test_files:
        save_path = os.path.join(args.output_dir, f"{file_key}_comparison.png")

        try:
            visualize_comparison(
                file_key=file_key,
                dxf_path=dxf_path,
                ours_model=ours_model,
                baseline_model=baseline_model,
                json_data=json_data,
                ours_metrics=ours_metrics,
                baseline_metrics=baseline_metrics,
                save_path=save_path,
                device=args.device,
            )
        except Exception as e:
            print(f"Error processing {file_key}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print(f"Comparison completed! Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
