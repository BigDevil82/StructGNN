"""
可视化分析：房间级预测 vs 墙段级预测的差异

展示为什么房间作为预测单元更合理：
1. 同一房间的四边墙体预测具有结构一致性
2. 相邻房间的剪力墙分布呈现对称性/连续性
3. 房间整体抗震等级对四边墙体的影响
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shearwall_pred.config import data_config, model_config
from src.shearwall_pred.cross_validate import EnsembleShearWallGNN
from src.shearwall_pred.dataset import ShearWallDataset


def visualize_room_coherence(
    dataset: ShearWallDataset, model: torch.nn.Module, sample_idx: int, save_path: Path
):
    """
    可视化房间级预测的一致性

    展示：
    1. 同一房间四边的预测是否协调
    2. 房间形状（方正/狭长）对剪力墙分布的影响
    3. 相邻房间之间的剪力墙连续性
    """
    device = next(model.parameters()).device
    data = dataset[sample_idx]

    # 模型预测 - 需要创建batch来处理单个样本
    model.eval()
    with torch.no_grad():
        # 使用Batch.from_data_list创建批处理，即使只有一个样本
        batch = Batch.from_data_list([data]).to(device)
        pred_prob, pred_ratio = model(batch)

    pred_prob = pred_prob.cpu().numpy()
    pred_ratio = pred_ratio.cpu().numpy()
    gt = data.y.cpu().numpy()

    # 提取房间信息
    rooms = data.x.cpu().numpy()  # (N, 25)
    num_rooms = rooms.shape[0]

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Room-level Prediction Coherence Analysis", fontsize=16, fontweight="bold")

    # 1. 房间形状分布
    ax = axes[0, 0]
    aspect_ratios = rooms[:, 4] / rooms[:, 5]  # 假设第5维和第6维是长和宽
    areas = rooms[:, 6]  # 假设第7维是面积
    scatter = ax.scatter(areas, aspect_ratios, c=gt.sum(axis=1), cmap="coolwarm", s=100, alpha=0.7)
    ax.set_xlabel("Room Area (m²)", fontsize=12)
    ax.set_ylabel("Aspect Ratio", fontsize=12)
    ax.set_title("Room Geometry vs. Total Wall Ratio", fontsize=13)
    plt.colorbar(scatter, ax=ax, label="Total Wall Ratio")

    # 2. 房间四边剪力墙分布的协调性
    ax = axes[0, 1]
    # 计算每个房间四边的剪力墙总比例
    edge_sums_gt = np.array(
        [
            (
                gt[i, 0:4].sum(),
                gt[i, 4:8].sum(),
                gt[i, 8:12].sum(),
                gt[i, 12:16].sum(),
            )
            for i in range(num_rooms)
        ]
    )
    edge_sums_pred = np.array(
        [
            (
                pred_ratio[i, 0:4].sum(),
                pred_ratio[i, 4:8].sum(),
                pred_ratio[i, 8:12].sum(),
                pred_ratio[i, 12:16].sum(),
            )
            for i in range(num_rooms)
        ]
    )

    edge_names = ["Top", "Right", "Bottom", "Left"]
    x = np.arange(4)
    width = 0.35

    # 随机选择3个代表性房间展示
    sample_rooms = np.random.choice(num_rooms, min(3, num_rooms), replace=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(sample_rooms)))

    for idx, (room_idx, color) in enumerate(zip(sample_rooms, colors)):
        offset = (idx - 1) * width / 2
        ax.plot(
            x + offset,
            edge_sums_gt[room_idx],
            "o-",
            label=f"GT Room {room_idx}",
            color=color,
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x + offset,
            edge_sums_pred[room_idx],
            "s--",
            label=f"Pred Room {room_idx}",
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(edge_names)
    ax.set_ylabel("Wall Ratio Sum", fontsize=12)
    ax.set_title("Four-Edge Coherence (GT vs Pred)", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. 房间类型对剪力墙分布的影响
    ax = axes[0, 2]
    # 假设第9维是房间类型（0=bedroom, 1=living, 2=kitchen, 3=bathroom）
    room_types = rooms[:, 6].astype(int) if rooms.shape[1] > 8 else np.zeros(num_rooms, dtype=int)
    type_names = ["Bedroom", "Living", "Kitchen", "Bathroom", "Other"]

    type_wall_ratios = []
    for room_type in range(5):
        mask = room_types == room_type
        if mask.sum() > 0:
            avg_ratio = gt[mask].mean()
            type_wall_ratios.append(avg_ratio)
        else:
            type_wall_ratios.append(0)

    bars = ax.bar(type_names, type_wall_ratios, color=plt.cm.Pastel1(np.linspace(0, 1, 5)))
    ax.set_ylabel("Average Wall Ratio", fontsize=12)
    ax.set_title("Wall Distribution by Room Type", fontsize=13)
    ax.set_ylim(0, max(type_wall_ratios) * 1.2 if max(type_wall_ratios) > 0 else 1)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 4. 预测误差的空间分布
    ax = axes[1, 0]
    errors = np.abs(pred_ratio - gt).mean(axis=1)
    scatter = ax.scatter(areas, aspect_ratios, c=errors, cmap="Reds", s=100, alpha=0.7)
    ax.set_xlabel("Room Area (m²)", fontsize=12)
    ax.set_ylabel("Aspect Ratio", fontsize=12)
    ax.set_title("Prediction Error Distribution", fontsize=13)
    plt.colorbar(scatter, ax=ax, label="MAE")

    # 5. 条件化生成的影响（如果有抗震等级信息）
    ax = axes[1, 1]
    # 假设条件向量的第一维是抗震等级（6度、7度、8度对应0, 1, 2）
    if hasattr(data, "condition"):
        seismic_level = data.condition[0].argmax().item() + 6
        title_suffix = f"(Seismic Level {seismic_level})"
    else:
        title_suffix = ""

    # 绘制GT和预测的整体分布直方图
    ax.hist(gt.flatten(), bins=50, alpha=0.5, label="Ground Truth", color="blue", density=True)
    ax.hist(pred_ratio.flatten(), bins=50, alpha=0.5, label="Prediction", color="red", density=True)
    ax.set_xlabel("Wall Ratio", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Wall Ratio Distribution {title_suffix}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 6. 房间间的剪力墙连续性（邻接房间的边界墙是否一致）
    ax = axes[1, 2]
    # 这需要边信息，简化展示：计算预测的标准差作为连续性指标
    if hasattr(data, "edge_index"):
        edge_index = data.edge_index.cpu().numpy()
        consistency_scores = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # 相邻房间的墙体预测应该相似
            consistency = 1 - np.abs(pred_ratio[src] - pred_ratio[dst]).mean()
            consistency_scores.append(consistency)

        ax.hist(consistency_scores, bins=30, color="green", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Consistency Score", fontsize=12)
        ax.set_ylabel("Number of Edges", fontsize=12)
        ax.set_title("Adjacent Room Wall Continuity", fontsize=13)
        ax.axvline(
            np.mean(consistency_scores),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(consistency_scores):.3f}",
        )
        ax.legend(fontsize=11)
    else:
        ax.text(
            0.5,
            0.5,
            "No edge information available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved room coherence visualization to {save_path}")

    # 返回统计数据
    stats = {
        "avg_room_area": float(areas.mean()),
        "avg_aspect_ratio": float(aspect_ratios.mean()),
        "avg_prediction_error": float(errors.mean()),
        "wall_ratio_by_room_type": {type_names[i]: float(type_wall_ratios[i]) for i in range(5)},
    }

    if hasattr(data, "edge_index"):
        stats["avg_adjacent_consistency"] = float(np.mean(consistency_scores))

    return stats


def analyze_room_level_advantages(
    dataset: ShearWallDataset, model: torch.nn.Module, num_samples: int = 20, output_dir: Path = None
):
    """
    批量分析房间级预测的优势

    生成论文用的统计数据和可视化
    """
    if output_dir is None:
        output_dir = Path("experiments/analysis/room_level_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    # 随机选择样本进行分析
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print(f"Analyzing {len(sample_indices)} samples for room-level advantages...")

    for idx in sample_indices:
        stats = visualize_room_coherence(dataset, model, idx, output_dir / f"sample_{idx:03d}_coherence.png")
        all_stats.append(stats)

    # 汇总统计
    summary = {
        "num_samples": len(all_stats),
        "avg_room_area": np.mean([s["avg_room_area"] for s in all_stats]),
        "avg_aspect_ratio": np.mean([s["avg_aspect_ratio"] for s in all_stats]),
        "avg_prediction_error": np.mean([s["avg_prediction_error"] for s in all_stats]),
    }

    # 保存汇总数据
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Room-level Analysis Summary:")
    print(f"{'='*60}")
    print(f"Average Room Area: {summary['avg_room_area']:.2f} m²")
    print(f"Average Aspect Ratio: {summary['avg_aspect_ratio']:.2f}")
    print(f"Average Prediction Error: {summary['avg_prediction_error']:.4f}")
    print(f"Results saved to {output_dir}")

    return summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize room-level prediction advantages")
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Path to cross-validation results directory"
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to analyze")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualizations")

    args = parser.parse_args()

    # 加载测试数据集
    print("Loading test dataset...")
    test_set = ShearWallDataset(
        root=f"{data_config.CACHE_DIR}/test", dxf_dir=f"{data_config.DXF_DIR}/test", is_test=True
    )
    print(f"Loaded {len(test_set)} test samples")

    # 加载K-Fold Ensemble模型
    print("\nLoading K-Fold Ensemble model...")
    cv_path = Path(args.result_dir)
    model_paths = sorted(cv_path.glob("fold_*/best_model.pth"))

    if len(model_paths) == 0:
        raise FileNotFoundError(f"No model checkpoints found in {cv_path}")

    print(f"Found {len(model_paths)} fold models")
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.eval()

    # 运行分析
    output_dir = Path(args.output_dir) if args.output_dir else None
    summary = analyze_room_level_advantages(
        test_set, ensemble_model, num_samples=args.num_samples, output_dir=output_dir
    )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
