"""
基于图像IoU的统一模型对比评估脚本

将Room-based和Edge-based模型的预测结果渲染为图像，计算像素级IoU进行公平对比。

使用方法:
    python experiments/metrics/compare_with_image_iou.py
    python experiments/metrics/compare_with_image_iou.py --ours_dir outputs/result/shearwall_pred/0126_cond_kfold --baseline_dir outputs/result/shearwall_pred/baseline-GNN-EP4-CV
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from shapely.geometry import LineString, Polygon
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============== 图像IoU计算 ==============
from docs.paper.experiments.metrics.image_iou import ImageIoUCalculator, edge_pred_to_walls, vector_to_walls

# ============== Baseline模型相关导入 ==============
from docs.paper.experiments.research.baseline_edge_gnn.config import data_config as baseline_data_config
from docs.paper.experiments.research.baseline_edge_gnn.config import model_config as baseline_model_config
from docs.paper.experiments.research.baseline_edge_gnn.config import train_config as baseline_train_config
from docs.paper.experiments.research.baseline_edge_gnn.ensemble_model import EnsembleGNNEP4
from docs.paper.experiments.research.baseline_edge_gnn.graph_builder import (
    build_graph_from_json,
    load_json_data,
)

# ============== 本研究模型相关导入 ==============
from src.shearwall_pred.config import data_config as ours_data_config
from src.shearwall_pred.config import model_config as ours_model_config
from src.shearwall_pred.config import training_config as ours_training_config
from src.shearwall_pred.cross_validate import EnsembleShearWallGNN
from src.shearwall_pred.dataset import ShearWallDataset
from src.shearwall_pred.utils import build_graph_from_dxf


def get_file_keys_from_dir(dir_path: str) -> List[str]:
    """从目录中获取DXF文件名（不含扩展名）"""
    file_keys = []
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dxf"):
                key = os.path.splitext(fname)[0]
                file_keys.append(key)
    return file_keys


@torch.no_grad()
def evaluate_ours_with_image_iou(
    model,
    test_dir: str,
    cache_dir: str,
    device,
    image_size: Tuple[int, int] = (512, 512),
) -> Tuple[Dict, List[Dict]]:
    """
    使用图像IoU评估Room-based模型

    Returns:
        aggregate_results: 汇总结果
        per_sample_results: 每个样本的结果
    """
    model.eval()
    calc = ImageIoUCalculator(image_size=image_size, line_width=3)

    # 获取测试文件
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".dxf")])

    per_sample_results = []
    all_ious = []

    # 用于全局指标计算（Micro平均）
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_mae_sum, total_mae_count = 0.0, 0
    total_se_sum = 0.0  # squared error for RMSE

    for fname in tqdm(test_files, desc="Evaluating Room-based"):
        file_path = os.path.join(test_dir, fname)
        file_key = os.path.splitext(fname)[0]

        try:
            # 构建图数据
            from torch_geometric.data import Batch

            builder = build_graph_from_dxf(file_path, mode="none")
            data = builder.to_pyg_data()
            batch = Batch.from_data_list([data]).to(device)

            # 模型预测
            pred_prob, pred_ratio = model(batch)
            pred_prob = pred_prob.cpu().numpy()
            pred_ratio = pred_ratio.cpu().numpy()
            gt = data.y.cpu().numpy()

            # 组合预测结果
            pred_combined = (pred_prob > 0.5) * pred_ratio
            pred_combined = np.where(pred_combined < 0.01, 0.0, pred_combined)

            # 收集所有房间的墙体
            all_gt_walls = []
            all_pred_walls = []

            node_ids = list(builder.graph.nodes())
            for i, node_id in enumerate(node_ids):
                node_data = builder.graph.nodes[node_id]
                room_poly = node_data["poly"]

                gt_walls = vector_to_walls(room_poly, gt[i])
                pred_walls = vector_to_walls(room_poly, pred_combined[i])

                all_gt_walls.extend(gt_walls)
                all_pred_walls.extend(pred_walls)

            # 计算图像IoU
            iou, details = calc.compute_iou(all_gt_walls, all_pred_walls)

            # 计算分类指标（向量级） - 累加到全局统计
            gt_binary = (gt > 0.01).astype(int).flatten()
            pred_binary = (pred_combined > 0.01).astype(int).flatten()

            tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
            fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
            fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
            tn = ((pred_binary == 0) & (gt_binary == 0)).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # 计算MAE/RMSE（仅在有墙位置）- 累加到全局统计
            valid_mask = gt > 0.01
            if valid_mask.sum() > 0:
                diff = pred_combined[valid_mask] - gt[valid_mask]
                mae_sum = np.abs(diff).sum()
                se_sum = (diff**2).sum()
                mae_count = valid_mask.sum()
                total_mae_sum += mae_sum
                total_se_sum += se_sum
                total_mae_count += mae_count
                sample_mae = mae_sum / mae_count
                sample_rmse = np.sqrt(se_sum / mae_count)
            else:
                sample_mae = 0.0
                sample_rmse = 0.0

            all_ious.append(iou)

            # 样本级指标（用于保存详细结果）
            sample_prec = tp / (tp + fp + 1e-8)
            sample_rec = tp / (tp + fn + 1e-8)
            sample_f1 = 2 * sample_prec * sample_rec / (sample_prec + sample_rec + 1e-8)
            sample_acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

            per_sample_results.append(
                {
                    "file_key": file_key,
                    "image_iou": float(iou),
                    "precision": float(sample_prec),
                    "recall": float(sample_rec),
                    "f1": float(sample_f1),
                    "accuracy": float(sample_acc),
                    "mae": float(sample_mae),
                    "rmse": float(sample_rmse),
                }
            )

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    # 计算全局Micro平均指标
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-8)
    mae = total_mae_sum / (total_mae_count + 1e-8)
    rmse = np.sqrt(total_se_sum / (total_mae_count + 1e-8))

    aggregate_results = {
        "model": "Ours (Room-based GNN)",
        "image_iou": float(np.mean(all_ious)),
        "image_iou_std": float(np.std(all_ious)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mae": float(mae),
        "rmse": float(rmse),
        "num_samples": len(all_ious),
    }

    return aggregate_results, per_sample_results


@torch.no_grad()
def evaluate_baseline_with_image_iou(
    model,
    json_data: Dict,
    test_keys: List[str],
    device,
    image_size: Tuple[int, int] = (512, 512),
) -> Tuple[Dict, List[Dict]]:
    """
    使用图像IoU评估Edge-based模型

    Returns:
        aggregate_results: 汇总结果
        per_sample_results: 每个样本的结果
    """
    model.eval()
    calc = ImageIoUCalculator(image_size=image_size, line_width=3)

    per_sample_results = []
    all_ious = []

    # 用于全局指标计算（Micro平均）
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_mae_sum, total_mae_count = 0.0, 0
    total_se_sum = 0.0  # squared error for RMSE

    from torch_geometric.data import Batch

    for file_key in tqdm(test_keys, desc="Evaluating Edge-based"):
        try:
            # 构建图数据
            builder = build_graph_from_json(json_data, file_key, mode="none")
            if builder is None:
                continue

            data = builder.to_pyg_data()
            if data is None:
                continue

            batch = Batch.from_data_list([data]).to(device)

            # 模型预测
            pred = model(batch)
            pred = pred.cpu().numpy()
            pred = np.where(pred < 0.1, 0.0, pred)
            gt = data.y.cpu().numpy()
            edge_mask = data.edge_mask.cpu().numpy()

            # 收集所有墙体
            all_gt_walls = []
            all_pred_walls = []

            edge_idx = 0
            for edge_info in builder.edges:
                if edge_idx >= len(pred):
                    break

                edge_line = edge_info["line"]
                edge_type = edge_info["edge_type"]

                # 只处理PSW类型的边
                if edge_type == baseline_data_config.EDGE_TYPE_PSW:
                    ratio_start_gt, ratio_end_gt = gt[edge_idx]
                    ratio_start_pred, ratio_end_pred = pred[edge_idx]

                    gt_walls = edge_pred_to_walls(edge_line, ratio_start_gt, ratio_end_gt)
                    pred_walls = edge_pred_to_walls(edge_line, ratio_start_pred, ratio_end_pred)

                    all_gt_walls.extend(gt_walls)
                    all_pred_walls.extend(pred_walls)

                edge_idx += 2  # Bidirectional edges

            # 计算图像IoU
            iou, details = calc.compute_iou(all_gt_walls, all_pred_walls)

            # 计算分类指标（仅在PSW边上） - 累加到全局统计
            mask_bool = edge_mask.astype(bool)
            pred_masked = pred[mask_bool]
            gt_masked = gt[mask_bool]

            gt_binary = (gt_masked > 0.1).astype(int).flatten()
            pred_binary = (pred_masked > 0.1).astype(int).flatten()

            tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
            fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
            fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
            tn = ((pred_binary == 0) & (gt_binary == 0)).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # MAE/RMSE - 累加到全局统计
            diff = pred_masked - gt_masked
            mae_sum = np.abs(diff).sum()
            se_sum = (diff**2).sum()
            mae_count = pred_masked.size
            total_mae_sum += mae_sum
            total_se_sum += se_sum
            total_mae_count += mae_count
            sample_mae = mae_sum / (mae_count + 1e-8)
            sample_rmse = np.sqrt(se_sum / (mae_count + 1e-8))

            all_ious.append(iou)

            # 样本级指标（用于保存详细结果）
            sample_prec = tp / (tp + fp + 1e-8)
            sample_rec = tp / (tp + fn + 1e-8)
            sample_f1 = 2 * sample_prec * sample_rec / (sample_prec + sample_rec + 1e-8)
            sample_acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

            per_sample_results.append(
                {
                    "file_key": file_key,
                    "image_iou": float(iou),
                    "precision": float(sample_prec),
                    "recall": float(sample_rec),
                    "f1": float(sample_f1),
                    "accuracy": float(sample_acc),
                    "mae": float(sample_mae),
                    "rmse": float(sample_rmse),
                }
            )

        except Exception as e:
            print(f"Error processing {file_key}: {e}")
            continue

    # 计算全局Micro平均指标
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-8)
    mae = total_mae_sum / (total_mae_count + 1e-8)
    rmse = np.sqrt(total_se_sum / (total_mae_count + 1e-8))

    aggregate_results = {
        "model": "Baseline (Edge-based GNN)",
        "image_iou": float(np.mean(all_ious)),
        "image_iou_std": float(np.std(all_ious)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mae": float(mae),
        "rmse": float(rmse),
        "num_samples": len(all_ious),
    }

    return aggregate_results, per_sample_results


def print_comparison_table(ours: Dict, baseline: Dict):
    """打印对比表格"""
    print("\n" + "=" * 85)
    print("模型性能对比 - 基于图像IoU (Model Performance Comparison - Image IoU)")
    print("=" * 85)

    print(f"\n{'Metric':<20} {'Ours (Room-based)':<25} {'Baseline (Edge-based)':<25} {'Diff':<15}")
    print("-" * 85)

    # Image IoU
    ours_iou = f"{ours['image_iou']:.4f} ± {ours['image_iou_std']:.4f}"
    baseline_iou = f"{baseline['image_iou']:.4f} ± {baseline['image_iou_std']:.4f}"
    diff_iou = ours["image_iou"] - baseline["image_iou"]
    diff_str = f"+{diff_iou:.4f}" if diff_iou > 0 else f"{diff_iou:.4f}"
    print(f"{'Image IoU':<20} {ours_iou:<25} {baseline_iou:<25} {diff_str:<15}")

    # 其他指标
    for metric in ["precision", "recall", "f1", "accuracy"]:
        ours_val = f"{ours[metric]:.4f}"
        baseline_val = f"{baseline[metric]:.4f}"
        diff = ours[metric] - baseline[metric]
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{metric.capitalize():<20} {ours_val:<25} {baseline_val:<25} {diff_str:<15}")

    # MAE/RMSE (越小越好)
    for metric in ["mae", "rmse"]:
        ours_val = f"{ours[metric]:.4f}"
        baseline_val = f"{baseline[metric]:.4f}"
        diff = baseline[metric] - ours[metric]  # 正数表示ours更好
        diff_str = f"↓{diff:.4f}" if diff > 0 else f"↑{-diff:.4f}"
        print(f"{metric.upper():<20} {ours_val:<25} {baseline_val:<25} {diff_str:<15}")

    print("-" * 85)
    print(f"{'Samples':<20} {ours['num_samples']:<25} {baseline['num_samples']:<25}")
    print("=" * 85)

    print("\n注: Image IoU通过将墙体渲染为图像后计算像素级IoU，实现不同表示方式的公平对比。")
    print("    正数差异表示Ours优于Baseline。MAE/RMSE: ↓表示Ours更优(误差更小)。")


def generate_latex_table(ours: Dict, baseline: Dict) -> str:
    """生成LaTeX表格"""

    def fmt(val, decimals=4):
        return f"{val:.{decimals}f}"

    def fmt_diff(val, higher_better=True):
        """格式化差异值，正数表示ours更好"""
        if higher_better:
            diff = val
            if diff > 0:
                return f"+{diff:.4f}"
            else:
                return f"{diff:.4f}"
        else:
            # 对于MAE/RMSE，负数表示ours更好
            if val > 0:
                return f"$\\downarrow${val:.4f}"
            else:
                return f"$\\uparrow${-val:.4f}"

    # 计算差异
    diff_iou = ours["image_iou"] - baseline["image_iou"]
    diff_prec = ours["precision"] - baseline["precision"]
    diff_rec = ours["recall"] - baseline["recall"]
    diff_f1 = ours["f1"] - baseline["f1"]
    diff_acc = ours["accuracy"] - baseline["accuracy"]
    diff_mae = baseline["mae"] - ours["mae"]  # 正数表示ours更好
    diff_rmse = baseline["rmse"] - ours["rmse"]

    # 比较哪个更好（用于加粗）
    ours_better_iou = ours["image_iou"] > baseline["image_iou"]
    ours_better_prec = ours["precision"] > baseline["precision"]
    ours_better_rec = ours["recall"] > baseline["recall"]
    ours_better_f1 = ours["f1"] > baseline["f1"]
    ours_better_acc = ours["accuracy"] > baseline["accuracy"]
    ours_better_mae = ours["mae"] < baseline["mae"]
    ours_better_rmse = ours["rmse"] < baseline["rmse"]

    def bold_if(val, condition):
        s = fmt(val)
        return f"\\textbf{{{s}}}" if condition else s

    table = r"""
\begin{table}[htbp]
\centering
\caption{Performance comparison between Room-based GNN (Ours) and Edge-based GNN (Baseline) using Image IoU}
\label{tab:model_comparison_image_iou}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{Ours} & \textbf{Baseline} & \textbf{Diff} \\
\midrule
"""

    # Image IoU
    table += f"Image IoU & {bold_if(ours['image_iou'], ours_better_iou)} & "
    table += f"{bold_if(baseline['image_iou'], not ours_better_iou)} & "
    table += f"{fmt_diff(diff_iou, True)} \\\\\n"

    # Precision
    table += f"Precision & {bold_if(ours['precision'], ours_better_prec)} & "
    table += f"{bold_if(baseline['precision'], not ours_better_prec)} & "
    table += f"{fmt_diff(diff_prec, True)} \\\\\n"

    # Recall
    table += f"Recall & {bold_if(ours['recall'], ours_better_rec)} & "
    table += f"{bold_if(baseline['recall'], not ours_better_rec)} & "
    table += f"{fmt_diff(diff_rec, True)} \\\\\n"

    # F1
    table += f"F1 & {bold_if(ours['f1'], ours_better_f1)} & "
    table += f"{bold_if(baseline['f1'], not ours_better_f1)} & "
    table += f"{fmt_diff(diff_f1, True)} \\\\\n"

    # Accuracy
    table += f"Accuracy & {bold_if(ours['accuracy'], ours_better_acc)} & "
    table += f"{bold_if(baseline['accuracy'], not ours_better_acc)} & "
    table += f"{fmt_diff(diff_acc, True)} \\\\\n"

    # MAE
    table += f"MAE & {bold_if(ours['mae'], ours_better_mae)} & "
    table += f"{bold_if(baseline['mae'], not ours_better_mae)} & "
    table += f"{fmt_diff(diff_mae, False)} \\\\\n"

    # RMSE
    table += f"RMSE & {bold_if(ours['rmse'], ours_better_rmse)} & "
    table += f"{bold_if(baseline['rmse'], not ours_better_rmse)} & "
    table += f"{fmt_diff(diff_rmse, False)} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\downarrow$ indicates lower is better for regression metrics (MAE, RMSE).
\end{tablenotes}
\end{table}
"""

    return table


def main():
    parser = argparse.ArgumentParser(description="基于图像IoU的统一模型对比评估")
    parser.add_argument(
        "--ours_dir",
        type=str,
        default="outputs/result/shearwall_pred/0126_cond_kfold",
        help="本研究模型的K-Fold结果目录",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="outputs/result/shearwall_pred/baseline-GNN-EP4-CV",
        help="Baseline模型的K-Fold结果目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/result/metrics/image_iou_comparison",
        help="结果输出目录",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="渲染图像尺寸",
    )

    args = parser.parse_args()

    print("=" * 85)
    print("统一模型对比评估 - 基于图像IoU")
    print("Room-based GNN vs Edge-based GNN")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    image_size = (args.image_size, args.image_size)

    # ============== 评估本研究模型 ==============
    print("\n" + "-" * 40)
    print("1. 评估本研究模型 (Room-based GNN)")
    print("-" * 40)

    # 加载Ensemble模型
    ours_cv_path = Path(args.ours_dir)
    ours_model_paths = sorted(list(ours_cv_path.glob("fold_*/best_model.pth")))

    if not ours_model_paths:
        print(f"   ❌ 错误: 在 {ours_cv_path} 下未找到fold_*/best_model.pth")
        sys.exit(1)

    print(f"   加载 {len(ours_model_paths)} 个fold模型")
    ours_model = EnsembleShearWallGNN(ours_model_paths, ours_model_config)
    ours_model.to(device)
    ours_model.eval()

    # 评估
    test_dir = f"{ours_data_config.DXF_DIR}/test"
    cache_dir = f"{ours_data_config.CACHE_DIR}/test"
    ours_results, ours_per_sample = evaluate_ours_with_image_iou(
        ours_model, test_dir, cache_dir, device, image_size
    )
    print(f"   ✓ 评估完成: Image IoU={ours_results['image_iou']:.4f}, F1={ours_results['f1']:.4f}")

    # ============== 评估Baseline模型 ==============
    print("\n" + "-" * 40)
    print("2. 评估Baseline模型 (Edge-based GNN)")
    print("-" * 40)

    # 加载JSON数据
    json_data = load_json_data(baseline_data_config.JSON_PATH)
    test_keys = get_file_keys_from_dir(baseline_data_config.TEST_DIR)
    print(f"   测试文件数: {len(test_keys)}")

    # 加载Ensemble模型
    baseline_cv_path = Path(args.baseline_dir)
    if not baseline_cv_path.exists():
        print(f"   ❌ 错误: Baseline模型目录不存在: {baseline_cv_path}")
        sys.exit(1)

    baseline_model = EnsembleGNNEP4.from_cv_dir(
        cv_dir=str(baseline_cv_path),
        node_in_dim=baseline_model_config.NODE_IN_DIM,
        edge_in_dim=baseline_model_config.EDGE_IN_DIM,
        gnn_out_dim=baseline_model_config.GNN_OUT_DIM,
        dropout=baseline_model_config.DROPOUT,
    )
    baseline_model.to(device)
    baseline_model.eval()

    # 评估
    baseline_results, baseline_per_sample = evaluate_baseline_with_image_iou(
        baseline_model, json_data, test_keys, device, image_size
    )
    print(f"   ✓ 评估完成: Image IoU={baseline_results['image_iou']:.4f}, F1={baseline_results['f1']:.4f}")

    # ============== 打印对比表格 ==============
    print_comparison_table(ours_results, baseline_results)

    # ============== 保存结果 ==============
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 汇总结果
    comparison_results = {
        "ours": ours_results,
        "baseline": baseline_results,
        "comparison": {
            "image_iou_diff": ours_results["image_iou"] - baseline_results["image_iou"],
            "f1_diff": ours_results["f1"] - baseline_results["f1"],
            "precision_diff": ours_results["precision"] - baseline_results["precision"],
            "recall_diff": ours_results["recall"] - baseline_results["recall"],
            "mae_diff": baseline_results["mae"] - ours_results["mae"],
        },
    }

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    # 每个样本的结果
    with open(output_dir / "ours_per_sample.json", "w", encoding="utf-8") as f:
        json.dump(ours_per_sample, f, indent=2, ensure_ascii=False)

    with open(output_dir / "baseline_per_sample.json", "w", encoding="utf-8") as f:
        json.dump(baseline_per_sample, f, indent=2, ensure_ascii=False)

    # LaTeX表格
    latex_table = generate_latex_table(ours_results, baseline_results)
    with open(output_dir / "comparison_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"\n📁 结果已保存至: {output_dir}")
    print(f"   - comparison_summary.json")
    print(f"   - ours_per_sample.json")
    print(f"   - baseline_per_sample.json")
    print(f"   - comparison_table.tex")
    print("\n" + "=" * 85)


if __name__ == "__main__":
    main()
