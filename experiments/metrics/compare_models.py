"""
模型对比评估脚本

对比本研究模型（Room-based GNN）与Baseline模型（Edge-based GNN）的性能差异。
两个模型的输出表示不同，因此使用各自的评估方式，最后汇总到统一的表格。

使用方法:
    python experiments/metrics/compare_models.py
    python experiments/metrics/compare_models.py --ours_dir outputs/result/shearwall_pred/0126_cond_kfold --baseline_dir outputs/result/shearwall_pred/baseline-GNN-EP4/cv_models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============== Baseline模型相关导入 ==============
from experiments.research.baseline_edge_gnn.config import data_config as baseline_data_config
from experiments.research.baseline_edge_gnn.config import model_config as baseline_model_config
from experiments.research.baseline_edge_gnn.config import train_config as baseline_train_config
from experiments.research.baseline_edge_gnn.dataset import EdgeShearWallDataset
from experiments.research.baseline_edge_gnn.ensemble_model import EnsembleGNNEP4

# ============== 本研究模型相关导入 ==============
from shearwall_pred.config import data_config as ours_data_config
from shearwall_pred.config import model_config as ours_model_config
from shearwall_pred.config import training_config as ours_training_config
from shearwall_pred.cross_validate import EnsembleShearWallGNN
from shearwall_pred.dataset import ShearWallDataset
from shearwall_pred.utils import calculate_vector_iou


def get_file_keys_from_dir(dir_path: str):
    """从目录中获取DXF文件名（不含扩展名）"""
    import os

    file_keys = []
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dxf"):
                key = os.path.splitext(fname)[0]
                file_keys.append(key)
    return file_keys


@torch.no_grad()
def evaluate_ours(model, test_loader, device) -> dict:
    """
    评估本研究模型（Room-based GNN）

    输出格式: 节点级 (N, 16) 的剪力墙比例向量
    评估指标: Vector IoU, Precision, Recall, F1, MAE
    """
    model.eval()

    all_pred_prob = []
    all_pred_ratio = []
    all_gt = []
    all_mask = []

    for batch in test_loader:
        batch = batch.to(device)
        pred_prob, pred_ratio = model(batch)

        all_pred_prob.append(pred_prob.cpu().numpy())
        all_pred_ratio.append(pred_ratio.cpu().numpy())
        all_gt.append(batch.y.cpu().numpy())
        all_mask.append(batch.constraint_mask.cpu().numpy())

    pred_prob = np.concatenate(all_pred_prob, axis=0)
    pred_ratio = np.concatenate(all_pred_ratio, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    # 1. 分类指标 (墙体存在性)
    prob_threshold = 0.5
    ratio_threshold = 0.01

    gt_binary = (gt > ratio_threshold).astype(int).flatten()
    pred_combined = (pred_prob > prob_threshold) * pred_ratio
    pred_binary = (pred_combined > ratio_threshold).astype(int).flatten()

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
    tn = ((pred_binary == 0) & (gt_binary == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    # 2. 回归指标 (墙体长度)
    valid_mask = gt > ratio_threshold
    if valid_mask.sum() > 0:
        mae = np.abs(pred_ratio[valid_mask] - gt[valid_mask]).mean()
        rmse = np.sqrt(((pred_ratio[valid_mask] - gt[valid_mask]) ** 2).mean())
    else:
        mae = 0.0
        rmse = 0.0

    # 3. Vector IoU
    iou = calculate_vector_iou(pred_combined, gt)

    return {
        "model": "Ours (Room-based GNN)",
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mae": float(mae),
        "rmse": float(rmse),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }


@torch.no_grad()
def evaluate_baseline(model, test_loader, device, threshold=0.1) -> dict:
    """
    评估Baseline模型（Edge-based GNN）

    输出格式: 边级 (E, 2) 的剪力墙比例 [ratio_start, ratio_end]
    评估指标: Precision, Recall, F1, MAE

    注意: Baseline模型没有直接的IoU计算，因为其表示方式不同
    """
    model.eval()

    all_pred = []
    all_target = []
    all_mask = []

    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)

        all_pred.append(pred.cpu().numpy())
        all_target.append(batch.y.cpu().numpy())
        all_mask.append(batch.edge_mask.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    edge_mask = np.concatenate(all_mask, axis=0)

    # 只在PSW边（可布置剪力墙的边）上计算指标
    mask_bool = edge_mask.astype(bool)
    pred_masked = pred[mask_bool]
    target_masked = target[mask_bool]

    # 1. MAE
    mae = np.abs(pred_masked - target_masked).mean()
    rmse = np.sqrt(((pred_masked - target_masked) ** 2).mean())

    # 2. 分类指标
    pred_binary = (pred_masked > threshold).astype(int).flatten()
    target_binary = (target_masked > threshold).astype(int).flatten()

    tp = ((pred_binary == 1) & (target_binary == 1)).sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum()
    tn = ((pred_binary == 0) & (target_binary == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "model": "Baseline (Edge-based GNN)",
        "iou": None,  # Baseline没有直接的IoU计算
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "mae": float(mae),
        "rmse": float(rmse),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }


def print_comparison_table(ours_results: dict, baseline_results: dict):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("模型性能对比 (Model Performance Comparison)")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Ours (Room-based)':<25} {'Baseline (Edge-based)':<25}")
    print("-" * 70)

    # IoU (只有我们的模型有)
    ours_iou = f"{ours_results['iou']:.4f}" if ours_results["iou"] else "N/A"
    baseline_iou = "N/A (不同表示)"
    print(f"{'Vector IoU':<20} {ours_iou:<25} {baseline_iou:<25}")

    # 分类指标
    for metric in ["precision", "recall", "f1", "accuracy"]:
        ours_val = f"{ours_results[metric]:.4f}"
        baseline_val = f"{baseline_results[metric]:.4f}"

        # 计算差异
        diff = ours_results[metric] - baseline_results[metric]
        if diff > 0:
            diff_str = f"(+{diff:.4f})"
        else:
            diff_str = f"({diff:.4f})"

        print(f"{metric.capitalize():<20} {ours_val:<25} {baseline_val:<25} {diff_str}")

    # 回归指标
    for metric in ["mae", "rmse"]:
        ours_val = f"{ours_results[metric]:.4f}"
        baseline_val = f"{baseline_results[metric]:.4f}"

        # MAE/RMSE越小越好
        diff = baseline_results[metric] - ours_results[metric]
        if diff > 0:
            diff_str = f"(↓{diff:.4f})"
        else:
            diff_str = f"(↑{-diff:.4f})"

        print(f"{metric.upper():<20} {ours_val:<25} {baseline_val:<25} {diff_str}")

    print("-" * 70)
    print("\n注: Baseline模型使用边级表示，无法直接计算Vector IoU。")
    print("    正数差异表示Ours优于Baseline，负数表示Baseline更优。")
    print("    MAE/RMSE: ↓表示Ours更优(误差更小)，↑表示Baseline更优。")


def main():
    parser = argparse.ArgumentParser(description="模型对比评估")
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
        default="outputs/result/metrics/comparison_results",
        help="结果输出目录",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("模型对比评估 - Room-based GNN vs Edge-based GNN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # ============== 评估本研究模型 ==============
    print("\n" + "-" * 40)
    print("1. 评估本研究模型 (Room-based GNN)")
    print("-" * 40)

    # 加载测试集
    ours_test_set = ShearWallDataset(
        f"{ours_data_config.CACHE_DIR}/test",
        f"{ours_data_config.DXF_DIR}/test",
        is_test=True,
    )
    ours_test_loader = DataLoader(
        ours_test_set,
        batch_size=ours_training_config.BATCH_SIZE,
        shuffle=False,
    )
    print(f"   测试集样本数: {len(ours_test_set)}")

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
    ours_results = evaluate_ours(ours_model, ours_test_loader, device)
    print(f"   ✓ 评估完成: IoU={ours_results['iou']:.4f}, F1={ours_results['f1']:.4f}")

    # ============== 评估Baseline模型 ==============
    print("\n" + "-" * 40)
    print("2. 评估Baseline模型 (Edge-based GNN)")
    print("-" * 40)

    # 加载测试集
    test_keys = get_file_keys_from_dir(baseline_data_config.TEST_DIR)
    print(f"   测试文件数: {len(test_keys)}")

    baseline_test_set = EdgeShearWallDataset(
        root="data/cache/baseline_comparison_test",
        json_path=baseline_data_config.JSON_PATH,
        file_keys=test_keys,
        is_test=True,
    )
    baseline_test_loader = DataLoader(
        baseline_test_set,
        batch_size=baseline_train_config.BATCH_SIZE,
        shuffle=False,
    )
    print(f"   测试集样本数: {len(baseline_test_set)}")

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
    baseline_results = evaluate_baseline(baseline_model, baseline_test_loader, device)
    print(f"   ✓ 评估完成: F1={baseline_results['f1']:.4f}, MAE={baseline_results['mae']:.4f}")

    # ============== 打印对比表格 ==============
    print_comparison_table(ours_results, baseline_results)

    # ============== 保存结果 ==============
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    comparison_results = {
        "ours": ours_results,
        "baseline": baseline_results,
        "comparison": {
            "f1_diff": ours_results["f1"] - baseline_results["f1"],
            "precision_diff": ours_results["precision"] - baseline_results["precision"],
            "recall_diff": ours_results["recall"] - baseline_results["recall"],
            "mae_diff": baseline_results["mae"] - ours_results["mae"],  # 正数表示ours更好
        },
    }

    output_file = output_dir / "model_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 结果已保存至: {output_file}")

    # 生成LaTeX表格
    latex_table = generate_latex_table(ours_results, baseline_results)
    latex_file = output_dir / "comparison_table.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"📁 LaTeX表格已保存至: {latex_file}")

    print("\n" + "=" * 80)


def generate_latex_table(ours: dict, baseline: dict) -> str:
    """生成LaTeX格式的对比表格"""

    def fmt(val, best=False):
        if val is None:
            return "-"
        s = f"{val:.4f}"
        if best:
            return f"\\textbf{{{s}}}"
        return s

    # 比较哪个更好
    ours_better_f1 = ours["f1"] > baseline["f1"]
    ours_better_prec = ours["precision"] > baseline["precision"]
    ours_better_rec = ours["recall"] > baseline["recall"]
    ours_better_mae = ours["mae"] < baseline["mae"]

    table = r"""
\begin{table}[htbp]
\centering
\caption{Performance comparison between Room-based GNN (Ours) and Edge-based GNN (Baseline)}
\label{tab:model_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{MAE} & \textbf{IoU} \\
\midrule
"""

    # Baseline行
    table += f"Baseline (Edge-based) & {fmt(baseline['precision'], not ours_better_prec)} & "
    table += f"{fmt(baseline['recall'], not ours_better_rec)} & "
    table += f"{fmt(baseline['f1'], not ours_better_f1)} & "
    table += f"{fmt(baseline['mae'], not ours_better_mae)} & - \\\\\n"

    # Ours行
    table += f"\\textbf{{Ours (Room-based)}} & {fmt(ours['precision'], ours_better_prec)} & "
    table += f"{fmt(ours['recall'], ours_better_rec)} & "
    table += f"{fmt(ours['f1'], ours_better_f1)} & "
    table += f"{fmt(ours['mae'], ours_better_mae)} & {fmt(ours['iou'])} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return table


if __name__ == "__main__":
    main()
