"""
条件化生成2x2布局可视化脚本

将一个输入在不同条件下的生成结果绘制在2x2布局中：
- 左上角：Ground Truth（工程师设计）
- 右上角：Group7-H1 (条件0)
- 左下角：Group7-H2 (条件1)
- 右下角：Group8 (条件2)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocess.room_analyzer import plot_room_analysis, reconstruct_walls
from shearwall_pred.config import data_config, model_config, training_config, viz_config
from shearwall_pred.cross_validate import EnsembleShearWallGNN
from shearwall_pred.utils import build_graph_from_dxf


def get_category_label(category: int) -> str:
    """获取类别标签"""
    labels = {
        0: "Group7-H1",
        1: "Group7-H2",
        2: "Group8",
    }
    return labels.get(category, f"Category {category}")


def visualize_conditional_grid(
    dxf_path: str,
    model,
    save_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    在1x3网格中可视化条件化生成结果

    布局：
    ┌─────────────┬─────────────┬─────────────┐
    │  Group7-H1  │  Group7-H2  │   Group8    │
    │(Condition 0)│(Condition 1)│(Condition 2)│
    └─────────────┴─────────────┴─────────────┘
    """
    # 构建图数据
    builder = build_graph_from_dxf(dxf_path, mode="none")
    aspect_ratio = 0.5

    # 根据宽高比动态设置figsize
    # 基准宽度为每个子图5英寸
    base_width = 8
    fig_width = base_width * 3  # 3列
    fig_height = base_width * aspect_ratio  # 1行

    # 创建1x3子图
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    # 三个条件化生成结果
    conditions = [0, 1, 2]
    iou_scores_all = []

    for cond, ax in zip(conditions, axes):
        # 构建数据
        data = builder.to_pyg_data()
        data_batch = Batch.from_data_list([data]).to(device)

        # 创建条件one-hot向量
        num_conditions = 3
        condition_tensor = torch.zeros(1, num_conditions, device=device)
        condition_tensor[0, cond] = 1

        # 预测
        model.eval()
        with torch.no_grad():
            pred_prob, pred_ratio = model(data_batch, condition=condition_tensor)
            pred_combined = (pred_prob > viz_config.PRED_PROB_THRESHOLD) * pred_ratio
            predictions = pred_combined.cpu().numpy()
            predictions = np.where(predictions < viz_config.PRED_RATIO_THRESHOLD, 0.0, predictions)

        # 绘制预测结果
        node_ids = list(builder.graph.nodes())
        iou_scores = []

        for i, node_id in enumerate(node_ids):
            node_data = builder.graph.nodes[node_id]
            room_poly = node_data["poly"]
            masks = node_data.get("masks", [])
            gt_vec = node_data.get("sw_vector")
            pred_vec = predictions[i]

            # 重建墙体
            pred_walls = reconstruct_walls(room_poly, pred_vec, masks)
            gt_walls = reconstruct_walls(room_poly, gt_vec, masks)

            # 绘图
            plot_room_analysis(
                room_poly, pred_vec, masks, ax, walls=pred_walls, wall_color=viz_config.PRED_WALL_COLOR
            )

            # # 计算IoU
            # iou = calculate_wall_iou(gt_walls, pred_walls, buffer_width=viz_config.IOU_BUFFER_WIDTH)
            # iou_scores.append(iou)

        # 设置样式
        # avg_iou = np.mean(iou_scores) if iou_scores else 0
        # iou_scores_all.append(avg_iou)

        # condition_label = get_category_label(cond)
        ax.set_aspect("equal")
        ax.axis("off")
        # ax.set_title(f"Pred ({condition_label})", fontsize=16, fontweight="bold", pad=10)

    # 总标题
    # fig.suptitle(f"Conditional Generation: {file_key}", fontsize=18, fontweight="bold", y=0.98)

    # 调整子图间距：hspace控制垂直间距，wspace控制水平间距
    plt.subplots_adjust(hspace=0.15, wspace=0, left=0.05, right=0.95, top=0.95, bottom=0.05)

    if save_path:
        plt.savefig(save_path, dpi=viz_config.DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return iou_scores_all


def process_test_set_conditional(
    cv_dir: str,
    test_dir: str,
    output_dir: str,
    max_samples: int = None,
    device: str = "cuda",
):
    """在测试集上批量处理条件化生成的2x2可视化"""

    # 加载集成模型
    print("Loading ensemble model...")
    cv_path = Path(cv_dir)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))

    if not model_paths:
        raise FileNotFoundError(f"No fold_*/best_model.pth found in {cv_path}")

    print(f"Found {len(model_paths)} model checkpoints")
    model_paths_str = [str(p) for p in model_paths]

    ensemble_model = EnsembleShearWallGNN(model_paths_str, model_config)
    ensemble_model.to(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取测试文件
    test_files = []
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(".dxf"):
            test_files.append((fname, os.path.join(test_dir, fname)))

    if max_samples:
        test_files = test_files[:max_samples]

    print(f"\nProcessing {len(test_files)} test samples...")
    print("-" * 80)

    # 处理每个测试样本
    for fname, dxf_path in test_files:
        file_key = fname.replace(".dxf", "")
        save_path = os.path.join(output_dir, f"{file_key}_conditional_grid.png")

        try:
            print(f"Processing: {file_key}")
            iou_scores = visualize_conditional_grid(
                dxf_path=dxf_path,
                model=ensemble_model,
                save_path=save_path,
                device=device,
            )

        except Exception as e:
            print(f"Error processing {file_key}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 打印统计摘要
    print("\n" + "=" * 80)
    print("CONDITIONAL GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(test_files)}")

    print("=" * 80)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize conditional generation in 2x2 grid")
    parser.add_argument(
        "--cv_dir",
        type=str,
        default="outputs/result/shearwall_pred/0126_cond_kfold",
        help="Cross-validation models directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/dxf/shearwall_split_8_2/test",
        help="Test DXF files directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/result/shearwall_pred/conditional_grid_plots",
        help="Output directory for grid visualizations",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    process_test_set_conditional(
        cv_dir=args.cv_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
