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
from shapely.affinity import scale as _scale_geom
from shapely.geometry import LineString, Polygon
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


def _compute_symmetry_score(geom, axis_x: float) -> float:
    if geom is None or geom.is_empty or geom.area < 1e-6:
        return 0.0
    mirrored = _scale_geom(geom, xfact=-1.0, yfact=1.0, origin=(axis_x, 0.0))
    inter = geom.intersection(mirrored)
    inter_area = inter.area
    denom = 2.0 * geom.area - inter_area
    if denom <= 1e-9:
        return 0.0
    return float(inter_area / denom)


def detect_left_right_symmetry(
    geom,
    *,
    threshold: float = 0.85,
    search_ratio: float = 0.1,
    search_steps: int = 9,
    simplify_tol: float = 0.0,
):
    if geom is None or geom.is_empty:
        return False, None, 0.0

    if simplify_tol and simplify_tol > 0:
        geom = geom.simplify(simplify_tol)
        if geom.is_empty:
            return False, None, 0.0

    minx, _, maxx, _ = geom.bounds
    center = (minx + maxx) / 2.0
    span = max((maxx - minx) * search_ratio, 0.0)

    if search_steps <= 1 or span <= 1e-6:
        axes = [center]
    else:
        axes = np.linspace(center - span, center + span, num=search_steps)

    best_axis = center
    best_score = -1.0
    for axis_x in axes:
        score = _compute_symmetry_score(geom, axis_x)
        if score > best_score:
            best_score = score
            best_axis = float(axis_x)

    return best_score >= threshold, best_axis, best_score


def _mirror_line(line, axis_x: float) -> LineString:
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    p1 = coords[0]
    p2 = coords[-1]
    m1 = (2 * axis_x - p1[0], p1[1])
    m2 = (2 * axis_x - p2[0], p2[1])
    return LineString([m1, m2])


def _merge_intervals(intervals, gap: float = 0.0):
    if not intervals:
        return []
    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals if b - a > 1e-6)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(m[0], m[1]) for m in merged]


def _intersect_intervals(a, b):
    if not a or not b:
        return []
    a = _merge_intervals(a, gap=0.0)
    b = _merge_intervals(b, gap=0.0)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e - s > 1e-6:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _edge_info(line: LineString, tol: float):
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    if abs(y1 - y2) <= tol and abs(x1 - x2) > tol:
        y = (y1 + y2) / 2.0
        minx, maxx = (x1, x2) if x1 <= x2 else (x2, x1)
        return {
            "orientation": "h",
            "coord": y,
            "min": minx,
            "max": maxx,
            "line": LineString([(minx, y), (maxx, y)]),
        }
    if abs(x1 - x2) <= tol and abs(y1 - y2) > tol:
        x = (x1 + x2) / 2.0
        miny, maxy = (y1, y2) if y1 <= y2 else (y2, y1)
        return {
            "orientation": "v",
            "coord": x,
            "min": miny,
            "max": maxy,
            "line": LineString([(x, miny), (x, maxy)]),
        }
    return None


def _intervals_from_ratios(info, ratios):
    if not ratios:
        return []
    length = info["max"] - info["min"]
    if length <= 1e-6:
        return []
    out = []
    for r0, r1 in ratios:
        s = info["min"] + r0 * length
        e = info["min"] + r1 * length
        if e - s > 1e-6:
            out.append((s, e))
    return _merge_intervals(out, gap=0.0)


def _mirror_intervals(intervals, info_from, info_to, axis_x: float):
    if not intervals:
        return []
    if info_from["orientation"] == "h":
        mirrored = []
        for a, b in intervals:
            m1 = 2.0 * axis_x - b
            m2 = 2.0 * axis_x - a
            s, e = (m1, m2) if m1 <= m2 else (m2, m1)
            mirrored.append((s, e))
        return _clip_intervals(mirrored, info_to["min"], info_to["max"])

    return _clip_intervals(intervals, info_to["min"], info_to["max"])


def _clip_intervals(intervals, minv, maxv):
    out = []
    for a, b in intervals:
        s = max(minv, a)
        e = min(maxv, b)
        if e - s > 1e-6:
            out.append((s, e))
    return _merge_intervals(out, gap=0.0)


def _intervals_to_lines(info, intervals):
    lines = []
    for a, b in intervals:
        if b - a <= 1e-6:
            continue
        if info["orientation"] == "h":
            lines.append(LineString([(a, info["coord"]), (b, info["coord"])]))
        else:
            lines.append(LineString([(info["coord"], a), (info["coord"], b)]))
    return lines


def symmetrize_walls_on_skeleton(
    fem_builder,
    axis_x: float,
    *,
    mode: str = "union",
    snap: float = 10.0,
    min_overlap: float = 50.0,
):
    if not fem_builder.raw_walls or not fem_builder.room_edges:
        return fem_builder.raw_walls

    skeleton = fem_builder._merge_collinear_lines(fem_builder.room_edges)
    wall_lines = fem_builder._merge_collinear_lines(fem_builder.raw_walls)

    infos = []
    for line in skeleton:
        info = _edge_info(line, fem_builder.snap_tolerance)
        if info:
            infos.append(info)

    if not infos:
        return fem_builder.raw_walls

    h_map = {}
    v_map = {}
    for idx, info in enumerate(infos):
        key = int(round(info["coord"] / snap))
        if info["orientation"] == "h":
            h_map.setdefault(key, []).append(idx)
        else:
            v_map.setdefault(key, []).append(idx)

    intervals_map = {}
    for idx, info in enumerate(infos):
        ratios = fem_builder._find_overlapping_segments(info["line"], wall_lines)
        intervals_map[idx] = _intervals_from_ratios(info, ratios)

    processed = set()
    new_intervals = dict(intervals_map)

    for i, info in enumerate(infos):
        if i in processed:
            continue

        mirror_line = _mirror_line(info["line"], axis_x)
        mirror_info = _edge_info(mirror_line, fem_builder.snap_tolerance)
        if not mirror_info:
            continue

        if info["orientation"] == "h":
            key = int(round(mirror_info["coord"] / snap))
            candidates = h_map.get(key, [])
            best = None
            best_overlap = 0.0
            m_min, m_max = mirror_info["min"], mirror_info["max"]
            for j in candidates:
                if j == i:
                    continue
                c = infos[j]
                overlap = min(m_max, c["max"]) - max(m_min, c["min"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = j
            if best is None or best_overlap < min_overlap:
                continue
            j = best
        else:
            key = int(round(mirror_info["coord"] / snap))
            candidates = v_map.get(key, [])
            best = None
            best_overlap = 0.0
            m_min, m_max = mirror_info["min"], mirror_info["max"]
            for j in candidates:
                if j == i:
                    continue
                c = infos[j]
                overlap = min(m_max, c["max"]) - max(m_min, c["min"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = j
            if best is None or best_overlap < min_overlap:
                continue
            j = best

        if j in processed:
            continue

        processed.add(i)
        processed.add(j)

        intervals_i = intervals_map.get(i, [])
        intervals_j = intervals_map.get(j, [])
        intervals_j_m = _mirror_intervals(intervals_j, infos[j], infos[i], axis_x)

        if mode == "intersection":
            combined = _intersect_intervals(intervals_i, intervals_j_m)
        else:
            combined = _merge_intervals(intervals_i + intervals_j_m, gap=fem_builder.gap_tolerance)

        new_intervals[i] = combined
        new_intervals[j] = _mirror_intervals(combined, infos[i], infos[j], axis_x)

    new_walls = []
    for idx, info in enumerate(infos):
        intervals = new_intervals.get(idx, [])
        if not intervals:
            continue
        new_walls.extend(_intervals_to_lines(info, intervals))

    return new_walls


def run_case_study(
    dxf_path: str,
    model_dir: str,
    output_dir: str,
    category: int = None,
    device: str = "cuda",
    symmetry_mode: str = "none",
    symmetry_threshold: float = 0.85,
    symmetry_simplify: float = 0.0,
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

        print("  检测左右对称性...")
        is_sym, axis_x, score = detect_left_right_symmetry(
            infill_union,
            threshold=symmetry_threshold,
            search_ratio=symmetry_search_ratio,
            search_steps=symmetry_search_steps,
            simplify_tol=symmetry_simplify,
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
        fem_builder.raw_walls = symmetrize_walls_on_skeleton(
            fem_builder,
            symmetry_axis,
            mode=symmetry_mode,
            snap=symmetry_snap,
            min_overlap=fem_builder.min_length,
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
    parser.add_argument(
        "--symmetry_simplify",
        type=float,
        default=0.0,
        help="对称检测几何简化容差（mm），>0 可显著加速",
    )
    parser.add_argument(
        "--symmetry_grid",
        dest="symmetry_simplify",
        type=float,
        default=0.0,
        help="兼容参数：等同于 --symmetry_simplify",
    )
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
        symmetry_simplify=args.symmetry_simplify,
        symmetry_search_ratio=args.symmetry_search_ratio,
        symmetry_search_steps=args.symmetry_search_steps,
        symmetry_snap=args.symmetry_snap,
    )


if __name__ == "__main__":
    main()
