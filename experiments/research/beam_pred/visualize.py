import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from experiments.research.beam_pred.beam_dataset import BeamDataset
from experiments.research.beam_pred.train import train_pipeline


def plot_training_history(history, save_path=None):
    """绘制训练过程中的 Loss 和 Metrics 曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Loss 曲线
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax1.set_title("Training Loss over Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # 2. Metrics 曲线
    ax2.plot(epochs, history["val_precision"], "g--", label="Precision")
    ax2.plot(epochs, history["val_recall"], "orange", label="Recall")
    ax2.plot(epochs, history["val_f1"], "r-", linewidth=2, label="F1 Score")
    ax2.set_title("Validation Metrics over Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Score")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 训练曲线已保存至: {save_path}")
    else:
        plt.show()


def visualize_comparison(model, loader, device, save_dir, num_samples=5):
    """
    可视化预测结果对比
    上图：真实布置 (Ground Truth)
    下图：预测布置 (Prediction)
    """
    model.eval()
    count = 0

    # 颜色定义
    COLOR_SHEAR_WALL = "red"  # 剪力墙
    COLOR_INFILL_WALL = "gray"  # 填充墙 (稍微淡一点)
    COLOR_BEAM_TRUE = "blue"  # 真实的梁
    COLOR_BEAM_PRED = "green"  # 预测的梁
    COLOR_NODE = "black"

    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break
            batch = batch.to(device)

            # 预测
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.target_edge_index)
            preds = (torch.sigmoid(logits) > 0.5).long()

            # 获取坐标 (反归一化比较麻烦，直接画归一化后的坐标即可，形状是一样的)
            coords = batch.x.cpu().numpy()

            # 创建画布
            fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(12, 8))

            # --- 辅助函数：绘制底图 (墙体) ---
            def plot_background(ax):
                # 遍历所有物理边 (edge_index)
                edge_idx = batch.edge_index.cpu().numpy()
                edge_attr = batch.edge_attr.cpu().numpy()

                for i in range(edge_idx.shape[1]):
                    u, v = edge_idx[:, i]
                    attr = edge_attr[i]

                    # 检查类型: edge_attr 前5位是One-Hot
                    # 0: shear_wall, 1: infill_wall
                    is_shear = attr[0] == 1
                    is_infill = attr[1] == 1

                    if is_shear:
                        ax.plot(
                            [coords[u][0], coords[v][0]],
                            [coords[u][1], coords[v][1]],
                            color=COLOR_SHEAR_WALL,
                            linewidth=3,
                            alpha=0.6,
                            zorder=1,
                        )
                    elif is_infill:
                        ax.plot(
                            [coords[u][0], coords[v][0]],
                            [coords[u][1], coords[v][1]],
                            color=COLOR_INFILL_WALL,
                            linewidth=1,
                            alpha=0.3,
                            zorder=0,
                        )

                # 绘制节点
                # ax.scatter(coords[:, 0], coords[:, 1], c=COLOR_NODE, s=10, zorder=5)

            # --- 1. 绘制真实值 (Ground Truth) ---
            plot_background(ax_gt)

            # 绘制真实的梁
            target_idx = batch.target_edge_index.cpu().numpy()
            labels = batch.y.cpu().numpy()

            for i in range(target_idx.shape[1]):
                if labels[i] == 1:  # 真实存在梁
                    u, v = target_idx[:, i]
                    ax_gt.plot(
                        [coords[u][0], coords[v][0]],
                        [coords[u][1], coords[v][1]],
                        color=COLOR_BEAM_TRUE,
                        linewidth=4,
                        label="True Beam" if i == 0 else "",
                    )

            ax_gt.set_title(f"Ground Truth (Sample {count})")
            ax_gt.set_aspect("equal")
            ax_gt.axis("off")

            # --- 2. 绘制预测值 (Prediction) ---
            plot_background(ax_pred)

            # 绘制预测的梁
            pred_labels = preds.cpu().numpy()

            for i in range(target_idx.shape[1]):
                if pred_labels[i] == 1:  # 预测有梁
                    u, v = target_idx[:, i]

                    # 用虚线或者稍微不同的颜色
                    ax_pred.plot(
                        [coords[u][0], coords[v][0]],
                        [coords[u][1], coords[v][1]],
                        color=COLOR_BEAM_PRED,
                        linewidth=4,
                        linestyle="-",
                        label="Pred Beam" if i == 0 else "",
                    )

            ax_pred.set_title(f"Prediction (Sample {count})")
            ax_pred.set_aspect("equal")
            ax_pred.axis("off")

            # 添加图例 (去重)
            from matplotlib.lines import Line2D

            custom_lines = [
                Line2D([0], [0], color=COLOR_SHEAR_WALL, lw=3),
                Line2D([0], [0], color=COLOR_BEAM_TRUE, lw=4),
                Line2D([0], [0], color=COLOR_BEAM_PRED, lw=4),
            ]
            ax_pred.legend(custom_lines, ["Shear Wall", "True Beam", "Pred Beam"])

            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{save_dir}/cmp_{count}.png", dpi=300)

            count += 1
            plt.close()


def main():
    # 1. 运行训练并获取模型和历史
    train_input_dir = r"data/dxf/beam_split_8_2/train"
    train_dxf_files = glob.glob(os.path.join(train_input_dir, "*.dxf"))

    print("开始训练...")
    SAVE_DIR = "outputs/result/beam_pred/0114_ckpt_1"
    model, history = train_pipeline(train_dxf_files, save_dir=SAVE_DIR)

    test_dxf_dir = r"data/dxf/beam_split_8_2/test"
    test_dxf_files = glob.glob(os.path.join(test_dxf_dir, "*.dxf"))
    test_dataset = BeamDataset("data/cache/beam_dataset/test", dxf_files=test_dxf_files, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 绘制训练曲线
    print("绘制训练指标曲线...")
    plot_training_history(history)

    # 3. 可视化对比结果
    print("可视化预测结果 (Red=剪力墙, Blue=真实梁, Green=预测梁)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_res_dir = os.path.join(SAVE_DIR, "test_results")
    os.makedirs(test_res_dir, exist_ok=True)
    visualize_comparison(model, test_loader, device, save_dir=test_res_dir, num_samples=15)


if __name__ == "__main__":
    main()
