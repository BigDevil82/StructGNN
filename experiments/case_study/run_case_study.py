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
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from torch_geometric.data import Batch

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# [CHANGE 1] 引入新的 FEMTopologyBuilder，同时保留旧文件中的导出和可视化工具
from experiments.case_study.fem_builder import FEMTopologyBuilder, export_to_json, visualize_fem_result
from preprocess.dxf_extractor import DXFExtractor
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


def _extract_infill_union(dxf_path: str):
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)
    polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]
    if not polys:
        return None
    return unary_union(polys).buffer(0)


def _compute_symmetry_score(geom, axis_x: float, step: float) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    minx, miny, maxx, maxy = geom.bounds
    if maxx - minx < 1e-3 or maxy - miny < 1e-3:
        return 0.0

    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)

    inter = 0
    union = 0
    for x in xs:
        for y in ys:
            p = Point(x, y)
            in_a = geom.covers(p)
            x_m = 2 * axis_x - x
            in_b = geom.covers(Point(x_m, y))
            if in_a or in_b:
                union += 1
                if in_a and in_b:
                    inter += 1

    return float(inter / union) if union > 0 else 0.0


def detect_left_right_symmetry(
    geom,
    *,
    step: float = 200.0,
    threshold: float = 0.85,
    search_ratio: float = 0.1,
    search_steps: int = 9,
):
    if geom is None or geom.is_empty:
        return False, None, 0.0

    minx, _, maxx, _ = geom.bounds
    center = (minx + maxx) / 2.0
    span = max((maxx - minx) * search_ratio, 0.0)

    if span < step or search_steps <= 1:
        axes = [center]
    else:
        axes = np.linspace(center - span, center + span, num=search_steps)

    best_axis = center
    best_score = -1.0
    for axis_x in axes:
        score = _compute_symmetry_score(geom, axis_x, step)
        if score > best_score:
            best_score = score
            best_axis = float(axis_x)

    return best_score >= threshold, best_axis, best_score


def _line_key(line, snap: float) -> tuple:
    coords = list(line.coords)
    if len(coords) < 2:
        return ()
    p1 = (round(coords[0][0] / snap), round(coords[0][1] / snap))
    p2 = (round(coords[-1][0] / snap), round(coords[-1][1] / snap))
    if p1 > p2:
        p1, p2 = p2, p1
    return (p1[0], p1[1], p2[0], p2[1])


def _mirror_line(line, axis_x: float) -> LineString:
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    p1 = coords[0]
    p2 = coords[-1]
    m1 = (2 * axis_x - p1[0], p1[1])
    m2 = (2 * axis_x - p2[0], p2[1])
    return LineString([m1, m2])


def symmetrize_lines(lines, axis_x: float, *, mode: str = "union", snap: float = 10.0):
    if not lines:
        return lines

    keys = []
    key_map = {}
    for line in lines:
        key = _line_key(line, snap)
        if not key:
            continue
        keys.append(key)
        key_map.setdefault(key, []).append(line)

    key_set = set(keys)
    result = []

    if mode == "intersection":
        for line in lines:
            m_key = _line_key(_mirror_line(line, axis_x), snap)
            if m_key in key_set:
                result.append(line)
        return result

    if mode == "union":
        result.extend(lines)
        added = set(key_set)
        for line in lines:
            m_line = _mirror_line(line, axis_x)
            m_key = _line_key(m_line, snap)
            if m_key and m_key not in added:
                result.append(m_line)
                added.add(m_key)
        return result

    return lines


def run_case_study(
    dxf_path: str,
    model_dir: str,
    output_dir: str,
    category: int = None,
    device: str = "cuda",
    symmetry_mode: str = "none",
    symmetry_threshold: float = 0.85,
    symmetry_grid: float = 200.0,
    symmetry_search_ratio: float = 0.1,
    symmetry_search_steps: int = 9,
    symmetry_snap: float = 10.0,
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

    symmetry_axis = None
    symmetry_score = 0.0
    if symmetry_mode != "none":
        infill_union = _extract_infill_union(dxf_path)
        if infill_union is None:
            infill_union = unary_union(room_polys).buffer(0)

        is_sym, axis_x, score = detect_left_right_symmetry(
            infill_union,
            step=symmetry_grid,
            threshold=symmetry_threshold,
            search_ratio=symmetry_search_ratio,
            search_steps=symmetry_search_steps,
        )

        if is_sym:
            symmetry_axis = axis_x
            symmetry_score = score
            print(
                f"  检测到左右对称 (score={symmetry_score:.3f}, axis_x={symmetry_axis:.2f}), "
                f"将使用 {symmetry_mode} 对称化"
            )
        else:
            print(f"  对称检测未通过 (best_score={score:.3f})，跳过对称化")

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

    if symmetry_axis is not None:
        fem_builder.raw_walls = symmetrize_lines(
            fem_builder.raw_walls, symmetry_axis, mode=symmetry_mode, snap=symmetry_snap
        )

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
    export_to_json(result, os.path.join(output_dir, f"{file_name}_fem_data.json"))
    export_shearwall_coords(result, os.path.join(output_dir, f"{file_name}_shearwall_coords.json"))

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
    parser.add_argument(
        "--symmetry_mode",
        type=str,
        default="none",
        choices=["none", "union", "intersection"],
        help="对称化策略 (none/union/intersection)",
    )
    parser.add_argument("--symmetry_threshold", type=float, default=0.85, help="对称检测阈值")
    parser.add_argument("--symmetry_grid", type=float, default=200.0, help="对称检测采样步长")
    parser.add_argument("--symmetry_search_ratio", type=float, default=0.1, help="轴搜索范围比例")
    parser.add_argument("--symmetry_search_steps", type=int, default=9, help="轴搜索步数")
    parser.add_argument("--symmetry_snap", type=float, default=10.0, help="对称线段匹配容差")

    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_case_study(
        dxf_path=args.dxf_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        category=args.category,
        device=args.device,
        symmetry_mode=args.symmetry_mode,
        symmetry_threshold=args.symmetry_threshold,
        symmetry_grid=args.symmetry_grid,
        symmetry_search_ratio=args.symmetry_search_ratio,
        symmetry_search_steps=args.symmetry_search_steps,
        symmetry_snap=args.symmetry_snap,
    )


if __name__ == "__main__":
    main()
