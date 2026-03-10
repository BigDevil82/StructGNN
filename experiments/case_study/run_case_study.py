"""
Engineering Case Study: 从建筑平面图到有限元建模

完整流程：
1. 读取DXF建筑平面图
2. 使用训练好的模型预测剪力墙布置
3. 解析预测结果为FEM构件（剪力墙+梁）- [UPDATED: 使用 Shapely 拓扑构建]
4. 可视化检查
5. 导出为ETABS可用的格式
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# [CHANGE 1] 引入新的 FEMTopologyBuilder，同时保留旧文件中的导出和可视化工具
from experiments.case_study.fem_builder import FEMTopologyBuilder, export_to_json, visualize_fem_result
from shearwall_pred.config import model_config, viz_config
from shearwall_pred.cross_validate import EnsembleShearWallGNN
from shearwall_pred.utils import build_graph_from_dxf


def load_ensemble_model(cv_dir: str, device: str = "cuda") -> EnsembleShearWallGNN:
    """加载K-Fold Ensemble模型"""
    cv_path = Path(cv_dir)
    model_paths = sorted(list(cv_path.glob("fold_*/best_model.pth")))

    if not model_paths:
        raise FileNotFoundError(f"在 {cv_path} 下未找到模型")

    print(f"加载 {len(model_paths)} 个fold模型...")
    ensemble_model = EnsembleShearWallGNN(model_paths, model_config)
    ensemble_model.to(device)
    ensemble_model.eval()
    return ensemble_model


def predict_shear_walls(model, data_batch, device: str = "cuda") -> np.ndarray:
    """使用模型预测剪力墙分布"""
    model.eval()
    with torch.no_grad():
        pred_prob, pred_ratio = model(data_batch)
        pred_combined = (pred_prob > viz_config.PRED_PROB_THRESHOLD) * pred_ratio
        predictions = pred_combined.cpu().numpy()
        predictions = np.where(predictions < viz_config.PRED_RATIO_THRESHOLD, 0.0, predictions)
    return predictions


def run_case_study(
    dxf_path: str,
    model_dir: str,
    output_dir: str,
    category: int = None,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    file_name = Path(dxf_path).stem

    print("=" * 60)
    print("Engineering Case Study (Shapely-Enhanced)")
    print("=" * 60)
    print(f"Input: {dxf_path}")
    print(f"Output: {output_dir}")

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    model = load_ensemble_model(model_dir, device)

    # 2. 读取DXF并构建图
    print("\n[2/5] 读取DXF并构建图...")
    builder_graph = build_graph_from_dxf(dxf_path, mode="none")
    data = builder_graph.to_pyg_data()

    if category is not None:
        cate_one_hot = np.zeros((1, 3))
        cate_one_hot[:, category] = 1.0
        data.condition = torch.tensor(cate_one_hot, dtype=torch.float)
        print(f"  设置建筑类别: {category}")

    data_batch = Batch.from_data_list([data]).to(device)

    # 提取房间信息
    room_polys = []
    masks_list = []
    node_ids = list(builder_graph.graph.nodes())

    for node_id in node_ids:
        node_data = builder_graph.graph.nodes[node_id]
        room_polys.append(node_data["poly"])
        masks_list.append(node_data.get("masks", []))

    print(f"  房间数量: {len(room_polys)}")

    # 3. 模型预测
    print("\n[3/5] 模型预测...")
    predictions = predict_shear_walls(model, data_batch, device)

    # # 保存原始预测
    # np.save(os.path.join(output_dir, f"{file_name}_predictions.npy"), predictions)

    # 4. 解析为FEM构件 (使用新的 Builder)
    print("\n[4/5] 解析为FEM构件 (Shapely Topology)...")

    # [CHANGE 2] 初始化新的 FEMTopologyBuilder
    # gap_tolerance: 两个墙段距离小于200mm时自动合并
    # min_length: 忽略小于100mm的碎片
    fem_builder = FEMTopologyBuilder(gap_tolerance=200.0, min_length=100.0)

    for i, room_poly in enumerate(room_polys):
        masks = masks_list[i] if masks_list else None
        # [CHANGE 3] 使用 builder.add_room
        fem_builder.add_room(room_poly, predictions[i], masks)

    # [CHANGE 4] 构建并获取结果
    result = fem_builder.build()

    print(f"  节点数量: {result['statistics']['num_nodes']}")
    print(f"  构件总数: {result['statistics']['num_members']}")
    print(f"    - 剪力墙: {result['statistics']['num_shearwalls']}")
    print(f"    - 梁: {result['statistics']['num_beams']}")

    # 5. 可视化和导出
    print("\n[5/5] 可视化和导出...")

    # 可视化FEM结果 (复用现有函数，因为结果格式已对其兼容)
    visualize_fem_result(
        result,
        room_polys,
        title=f"FEM Topology (Shapely) - {file_name}",
        save_path=os.path.join(output_dir, f"{file_name}_fem_members.png"),
    )

    # 绘制预测对比图
    visualize_prediction_comparison(
        builder_graph,
        predictions,
        room_polys,
        save_path=os.path.join(output_dir, f"{file_name}_prediction.png"),
    )

    # 导出数据
    # export_to_json(result, os.path.join(output_dir, f"{file_name}_fem_data.json"))
    # export_shearwall_coords(result, os.path.join(output_dir, f"{file_name}_shearwall_coords.json"))

    print("\n" + "=" * 60)
    print("案例研究完成！")
    return result


def visualize_prediction_comparison(builder, predictions, room_polys, save_path=None):
    """可视化预测结果对比"""
    from preprocess.room_analyzer import plot_room_analysis, reconstruct_walls

    # 设置绘图后端避免无头模式报错
    plt.switch_backend("Agg")

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 8))
    node_ids = list(builder.graph.nodes())

    # Ground Truth
    ax_gt.set_title("Ground Truth", fontsize=14, fontweight="bold")
    for node_id in node_ids:
        node_data = builder.graph.nodes[node_id]
        if node_data.get("sw_vector") is not None:
            plot_room_analysis(
                node_data["poly"],
                node_data["sw_vector"],
                node_data.get("masks", []),
                ax_gt,
                wall_color="green",
            )

    # Prediction
    ax_pred.set_title("Model Prediction", fontsize=14, fontweight="bold")
    for i, node_id in enumerate(node_ids):
        node_data = builder.graph.nodes[node_id]
        pred_vec = predictions[i]
        masks = node_data.get("masks", [])
        pred_walls = reconstruct_walls(node_data["poly"], pred_vec, masks)
        plot_room_analysis(node_data["poly"], pred_vec, masks, ax_pred, walls=pred_walls, wall_color="red")

    for ax in [ax_gt, ax_pred]:
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def export_shearwall_coords(result: dict, output_path: str):
    """导出简化的剪力墙坐标"""
    shearwall_coords = []
    for member in result["members"]:
        if member["type"] == "shearwall":
            shearwall_coords.append(
                [
                    list(member["start_coord"]),
                    list(member["end_coord"]),
                ]
            )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shearwall_coords, f, indent=2)
    print(f"  Exported coords to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Engineering Case Study")
    parser.add_argument("--dxf_path", type=str, required=True, help="输入DXF文件路径")
    parser.add_argument("--model_dir", type=str, default="result/shearwall_pred/0126_cond_kfold")
    parser.add_argument("--output_dir", type=str, default="result/case_study")
    parser.add_argument("--category", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_case_study(
        dxf_path=args.dxf_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        category=args.category,
        device=args.device,
    )


if __name__ == "__main__":
    main()
