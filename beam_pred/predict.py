import argparse
import glob
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Batch

from beam_pred.beam_dataset import builder_to_pyg_data
from beam_pred.model import BeamPredictorGNN
from preprocess.beam_ir_builder import StructuralGraphBuilder


def load_model(model_path, device):
    """加载训练好的模型"""
    # 确保这里的维度参数与训练时一致 (node=2, edge=6)
    # 如果你加了 Node Type，记得改为 6
    model = BeamPredictorGNN(node_in_dim=6, edge_in_dim=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_single_dxf(dxf_path, model, device):
    """处理单个 DXF 并返回预测结果"""
    print(f"正在处理: {dxf_path}")

    # 1. 解析 DXF
    builder = StructuralGraphBuilder(dxf_path)
    builder.process()

    # 2. 转换为 PyG Data
    data = builder_to_pyg_data(builder)
    if data is None:
        print("❌ 无法生成图数据 (可能无房间或无节点)")
        return None, None

    # 3. 放入 Device (伪装成一个 Batch)
    batch = Batch.from_data_list([data]).to(device)

    # 4. 推理
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.target_edge_index)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

    return batch, preds, builder


def visualize_prediction(batch, preds, builder: StructuralGraphBuilder, save_path=None):
    """
    可视化对比图 (使用原始 CAD 坐标)
    左：真实 (Ground Truth)
    右：预测 (Prediction)
    """
    # === 关键修改：获取原始坐标 ===
    # batch.x 是归一化的，builder.node_manager.nodes 是原始的
    original_nodes = builder.node_manager.nodes

    # 辅助函数：通过节点 ID 获取原始坐标
    def get_pos(node_idx):
        node = original_nodes[node_idx]
        return node.x, node.y

    target_idx = batch.target_edge_index.cpu().numpy()
    gt_labels = batch.y.cpu().numpy()
    pred_labels = preds.cpu().numpy()

    # 颜色定义
    COLOR_SHEAR = "#FF4444"  # 鲜红 (剪力墙)
    COLOR_INFILL = "#CCCCCC"  # 浅灰 (填充墙)
    COLOR_GT_BEAM = "#0000FF"  # 纯蓝 (真实梁)
    COLOR_PRED_BEAM = "#00AA00"  # 深绿 (预测梁)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # 画布调大一点

    # --- 辅助：绘制背景墙 (原始坐标) ---
    def plot_background(ax):
        edge_idx = batch.edge_index.cpu().numpy()
        edge_attr = batch.edge_attr.cpu().numpy()

        # 绘制墙体
        for i in range(edge_idx.shape[1]):
            u, v = edge_idx[:, i]
            x1, y1 = get_pos(u)
            x2, y2 = get_pos(v)

            # attr[0]=ShearWall, attr[1]=Infill (根据 one-hot 定义)
            if edge_attr[i][0] == 1:  # Shear Wall
                ax.plot([x1, x2], [y1, y2], c=COLOR_SHEAR, lw=3, alpha=0.6, zorder=1)
            elif edge_attr[i][1] == 1:  # Infill Wall
                ax.plot([x1, x2], [y1, y2], c=COLOR_INFILL, lw=1, alpha=0.5, zorder=0)

        # 绘制节点 (可选，点小一点)
        # xs = [n.x for n in original_nodes]
        # ys = [n.y for n in original_nodes]
        # ax.scatter(xs, ys, c='black', s=2, zorder=2)

        ax.set_aspect("equal")  # 关键：保持 CAD 比例
        ax.axis("off")  # 隐藏坐标轴刻度

    # --- 左图：真实情况 (Ground Truth) ---
    plot_background(ax1)
    ax1.set_title("Ground Truth VS Prediction", fontsize=14)

    for i in range(target_idx.shape[1]):
        if gt_labels[i] == 1:
            u, v = target_idx[:, i]
            x1, y1 = get_pos(u)
            x2, y2 = get_pos(v)
            ax1.plot([x1, x2], [y1, y2], c=COLOR_GT_BEAM, lw=4, zorder=10)

    # --- 右图：AI 预测 (Prediction) ---
    plot_background(ax2)
    # ax2.set_title("AI Prediction (Original CAD Coords)", fontsize=14)

    for i in range(target_idx.shape[1]):
        if pred_labels[i] == 1:
            u, v = target_idx[:, i]
            x1, y1 = get_pos(u)
            x2, y2 = get_pos(v)
            ax2.plot([x1, x2], [y1, y2], c=COLOR_PRED_BEAM, lw=4, zorder=10)

    # 绘制图例 - 放置在两图中间
    handles = [
        plt.Line2D([0], [0], color=COLOR_SHEAR, lw=3, label="Shear Wall"),
        plt.Line2D([0], [0], color=COLOR_GT_BEAM, lw=4, label="True Beam"),
        plt.Line2D([0], [0], color=COLOR_PRED_BEAM, lw=4, label="Predicted Beam"),
    ]

    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        frameon=True,
        fancybox=True,
        fontsize=12,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 原始坐标可视化已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="梁布置预测推理工具")
    parser.add_argument(
        "--model", type=str, default="checkpoints/beam_predictor_model.pth", help="模型权重路径"
    )
    parser.add_argument("--output", type=str, default="result/beam_pred", help="输出图片路径")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 找不到模型文件 {args.model}. 请先运行 train.py")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = load_model(args.model, device)

    # 2. 预测
    os.makedirs(args.output, exist_ok=True)
    test_dxf_dir = r"dxf/beam_split_8_2/test"
    test_dxf_files = glob.glob(os.path.join(test_dxf_dir, "*.dxf"))
    for dxf_path in test_dxf_files:
        batch, preds, builder = predict_single_dxf(dxf_path, model, device)

        save_path = os.path.join(args.output, f"{os.path.basename(dxf_path).replace('.dxf', '')}.png")

        # 3. 可视化
        if batch is not None:
            visualize_prediction(batch, preds, builder, save_path=save_path)
