from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union


def get_room_edges(room_poly: Polygon) -> List[LineString]:
    """获取房间的四条边 [上、右、下、左]"""
    minx, miny, maxx, maxy = room_poly.bounds
    tl, tr, bl, br = Point(minx, maxy), Point(maxx, maxy), Point(minx, miny), Point(maxx, miny)
    return [
        LineString([tl, tr]),  # Top: 从左往右
        LineString([tr, br]),  # Right: 从上往下
        LineString([bl, br]),  # Bottom: 从左往右
        LineString([tl, bl]),  # Left: 从上往下
    ]


class RoomAnalyzer:
    """分析房间的剪力墙分布和可布置区域"""

    def __init__(self, sw_polys: List[Polygon], infill_polys: List[Polygon]):
        """
        Args:
            sw_polys: 剪力墙多边形列表
            infill_polys: 填充墙多边形列表
        """
        self.sw_union = unary_union(sw_polys).buffer(0)
        self.buildable_union = unary_union(sw_polys + infill_polys).buffer(0)

    def _compute_intervals(
        self, edge: LineString, poly_union: Polygon, threshold: float = 0.1
    ) -> List[Tuple[float, float]]:
        """计算边与多边形集合的交集区间（归一化到 [0, 1]）"""
        intersection = edge.intersection(poly_union)
        if intersection.is_empty or edge.length == 0:
            return []

        # 统一处理不同类型的几何对象
        geoms = []
        if isinstance(intersection, LineString):
            geoms = [intersection]
        elif isinstance(intersection, MultiLineString):
            geoms = list(intersection.geoms)
        elif isinstance(intersection, (Point, Polygon)):
            # Point 或 Polygon 类型不适合作为线段区间，跳过
            return []
        else:
            # 其他类型（如 GeometryCollection）尝试提取 LineString
            if hasattr(intersection, "geoms"):
                geoms = [g for g in intersection.geoms if isinstance(g, LineString)]

        # 投影到边上并归一化
        intervals = []
        for geom in geoms:
            d1 = edge.project(Point(geom.coords[0]), normalized=True)
            d2 = edge.project(Point(geom.coords[-1]), normalized=True)
            start, end = sorted([d1, d2])
            if end - start > threshold:
                intervals.append((start, end))

        # 合并重叠区间
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _compute_anchor_ratios(self, intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
        """计算边两端的剪力墙比例"""
        if not intervals:
            return 0.0, 0.0
        # if len(intervals) == 1:
        #     if intervals[0][0] < 1e-3:
        #         return intervals[0][1], 0.0
        #     else:
        #         return 0.0, 1.0 - intervals[0][0]
        start_ratio = intervals[0][1] if intervals[0][0] < 1e-3 else 0.0
        end_ratio = 1.0 - intervals[-1][0] if intervals[-1][1] > 1.0 - 1e-3 else 0.0
        return start_ratio, end_ratio

    def process_room(self, room_poly: Polygon) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
        """
        处理单个房间，计算剪力墙向量和可布置区域掩码

        Returns:
            sw_vector: 16维向量 [4条边 × 2半边 × 2端点]
            masks: 4个列表，每条边的可布置区间
        """
        edges = get_room_edges(room_poly)
        sw_vector = []
        masks = []

        for edge in edges:
            # 将边分为两半
            p_mid = edge.interpolate(0.5, normalized=True)
            halves = [
                LineString([edge.coords[0], p_mid]),
                LineString([p_mid, edge.coords[-1]]),
            ]

            # 计算每半边的剪力墙比例
            for half in halves:
                sw_intervals = self._compute_intervals(half, self.sw_union)
                start_ratio, end_ratio = self._compute_anchor_ratios(sw_intervals)
                sw_vector.extend([start_ratio, end_ratio])

                # 计算可布置区间
                buildable_intervals = self._compute_intervals(half, self.buildable_union)
                masks.append(buildable_intervals)

        return np.array(sw_vector), masks


def _walls_from_vector(
    room_poly: Polygon, sw_vector: np.ndarray, masks: List[List[Tuple[float, float]]] = None
) -> MultiLineString:
    """
    从剪力墙向量重建墙体线段 (修复了掩码裁剪逻辑)

    Args:
        room_poly: 房间多边形
        sw_vector: 16维剪力墙向量
        masks: 列表长度应为8 (4条边x2半边)，每个元素为该半边上的可布置区间列表 [(s, e), ...]
    """
    edges = get_room_edges(room_poly)
    sw_ratios = sw_vector.reshape(4, 2, 2)  # [4条边, 2半边, 2端点]
    result_walls = []

    # 扁平化索引，用于追踪当前处理的是第几个半边 (总共0-7)
    half_edge_idx = 0

    for i, edge in enumerate(edges):
        # 将边分为两半
        p_mid = edge.interpolate(0.5, normalized=True)
        halves = [LineString([edge.coords[0], p_mid]), LineString([p_mid, edge.coords[-1]])]

        for j, half in enumerate(halves):
            # 1. 生成预测的原始墙体
            start_ratio, end_ratio = sw_ratios[i][j]
            current_half_walls = []

            # 生成起点墙体
            if start_ratio > 1e-3:
                w_start = half.interpolate(0, normalized=True)
                w_end = half.interpolate(start_ratio, normalized=True)
                current_half_walls.append(LineString([w_start, w_end]))

            # 生成终点墙体
            if end_ratio > 1e-3:
                w_start = half.interpolate(1.0 - end_ratio, normalized=True)
                w_end = half.interpolate(1.0, normalized=True)
                current_half_walls.append(LineString([w_start, w_end]))

            # 如果该半边没有预测墙体，跳过并更新索引
            if not current_half_walls:
                half_edge_idx += 1
                continue

            # 2. 应用掩码裁剪 (Mask Clipping)
            # 确保 masks 存在且索引未越界
            if masks:
                valid_intervals = masks[half_edge_idx]

                # 如果该半边有可布置区间信息
                if valid_intervals:
                    # 将区间转换为当前半边上的几何线段
                    mask_lines = []
                    for s, e in valid_intervals:
                        # s, e 是相对于 half 的归一化坐标
                        p_s = half.interpolate(s, normalized=True)
                        p_e = half.interpolate(e, normalized=True)
                        mask_lines.append(LineString([p_s, p_e]))

                    # 使用布尔运算求交集
                    pred_union = unary_union(current_half_walls)
                    mask_union = unary_union(mask_lines)
                    final_walls = pred_union.intersection(mask_union)

                    if not final_walls.is_empty:
                        if isinstance(final_walls, LineString):
                            result_walls.append(final_walls)
                        elif isinstance(final_walls, MultiLineString):
                            result_walls.extend(final_walls.geoms)
                else:
                    # 如果 valid_intervals 为空列表，说明该半边完全不可布置，不添加任何墙体
                    pass
            else:
                # 如果没有提供掩码，保留原始预测
                result_walls.extend(current_half_walls)

            # 更新半边计数器
            half_edge_idx += 1

    return MultiLineString(result_walls) if result_walls else MultiLineString([])


def extract_ground_truth_walls(room_poly: Polygon, sw_vector: np.ndarray) -> MultiLineString:
    """从训练数据的剪力墙向量提取真实墙体（无掩码裁剪）"""
    return _walls_from_vector(room_poly, sw_vector, masks=None)


def reconstruct_walls(
    room_poly: Polygon, sw_vector: np.ndarray, masks: List[List[Tuple[float, float]]]
) -> MultiLineString:
    """根据预测向量和掩码重建墙体（有掩码裁剪）"""
    return _walls_from_vector(room_poly, sw_vector, masks=masks)


def calculate_wall_iou(
    gt_walls: MultiLineString, pred_walls: MultiLineString, buffer_width: float = 0.5
) -> float:
    """计算两组墙体的 IoU（通过 buffer 转换为面积）"""
    if gt_walls.is_empty and pred_walls.is_empty:
        return 1.0
    if gt_walls.is_empty or pred_walls.is_empty:
        return 0.0

    # 使用 cap_style 和 join_style 来获得更一致的 buffer 结果
    gt_area = gt_walls.buffer(
        buffer_width, cap_style=2, join_style=2
    )  # cap_style=2: flat, join_style=2: mitre
    pred_area = pred_walls.buffer(buffer_width, cap_style=2, join_style=2)

    # 计算交集和并集
    intersection_area = gt_area.intersection(pred_area).area
    union_area = gt_area.area + pred_area.area - intersection_area  # 使用公式避免 union 操作的误差

    return intersection_area / union_area if union_area > 1e-6 else 0.0


def plot_room_analysis(
    room_poly: Polygon,
    sw_vector: np.ndarray,
    masks: List[List[Tuple[float, float]]],
    ax: plt.Axes,
    walls: MultiLineString = None,
    wall_color: str = "red",
):
    """
    绘制单个房间的分析结果

    Args:
        room_poly: 房间多边形
        sw_vector: 剪力墙向量
        masks: 可布置区域掩码
        ax: matplotlib Axes
        walls: 要绘制的墙体（如果为None则从sw_vector提取）
        wall_color: 墙体颜色
    """
    # 绘制房间轮廓
    x, y = room_poly.exterior.xy
    ax.plot(x, y, "b--", alpha=0.7, linewidth=1, zorder=2)

    # 获取房间的四条边
    minx, miny, maxx, maxy = room_poly.bounds
    tl, tr, bl, br = Point(minx, maxy), Point(maxx, maxy), Point(minx, miny), Point(maxx, miny)
    edges = [LineString([tl, tr]), LineString([tr, br]), LineString([bl, br]), LineString([tl, bl])]

    # 绘制可布置区域（灰色）
    half_idx = 0
    for i, edge in enumerate(edges):
        # 将边分为两半
        p_mid = edge.interpolate(0.5, normalized=True)
        halves = [LineString([edge.coords[0], p_mid]), LineString([p_mid, edge.coords[-1]])]
        for half in halves:
            if half_idx < len(masks):
                for start, end in masks[half_idx]:
                    p1, p2 = half.interpolate(start, normalized=True), half.interpolate(end, normalized=True)
                    ax.plot([p1.x, p2.x], [p1.y, p2.y], color="#BBBBBB", linewidth=8, zorder=3)
            half_idx += 1

    # 绘制剪力墙
    if walls is None:
        walls = extract_ground_truth_walls(room_poly, sw_vector)
    if not walls.is_empty:
        geoms = walls.geoms if hasattr(walls, "geoms") else [walls]
        for wall in geoms:
            x, y = wall.xy
            ax.plot(x, y, color=wall_color, linewidth=4, zorder=4)


def plot_all_rooms(room_polys: List[Polygon], analysis_results: List[dict], save_path: str = None):
    """
    绘制所有房间的概览图

    Args:
        room_polys: 房间多边形列表
        analysis_results: 分析结果列表，每项包含 {'room_index', 'sw_vector', 'masks'}
        save_path: 保存路径（如果为None则显示）
    """
    if not room_polys:
        print("警告: 没有可供绘制的房间。")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    results_map = {res["room_index"]: res for res in analysis_results}

    for i, room_poly in enumerate(room_polys):
        if i in results_map:
            res = results_map[i]
            plot_room_analysis(room_poly, np.array(res["sw_vector"]), res["masks"], ax)
        else:
            # 无效房间只绘制轮廓
            x, y = room_poly.exterior.xy
            ax.plot(x, y, "gray", linestyle="--", linewidth=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("房间剪力墙和可布置区域分析")
    ax.set_xlabel("X (坐标)")
    ax.set_ylabel("Y (坐标)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"🖼️ 可视化结果已保存到: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    from dxf_extractor import DXFExtractor
    from preprocess.room_calibrator import calibrate_rooms

    # 提取DXF数据
    dxf_path = r"dxf/to_process/room_finished/L1L28_232.dxf"
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    # 转换为Shapely几何体
    sw_polys = [Polygon([(p.x, p.y) for p in poly]) for poly in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in poly]) for poly in extractor.infill_walls]
    room_polys = [Polygon([(p.x, p.y) for p in room.polygon]) for room in extractor.rooms]
    calibrated_room_polys = calibrate_rooms(room_polys, 200)

    # 初始化分析器
    analyzer = RoomAnalyzer(sw_polys, infill_polys)

    # 测试：对比 Ground Truth 和 Prediction
    print("\n" + "=" * 60)
    print("测试: Ground Truth vs Prediction 可视化")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction (全1.0向量)")

    iou_list = []

    for room_index, room_poly in enumerate(room_polys):
        if not room_poly.is_valid:
            continue

        # 获取真实数据
        gt_sw_vector, gt_masks = analyzer.process_room(room_poly)

        calibrated_room_poly = calibrated_room_polys[room_index]
        gt_walls = extract_ground_truth_walls(calibrated_room_poly, gt_sw_vector)
        plot_room_analysis(calibrated_room_poly, gt_sw_vector, gt_masks, ax1, gt_walls, wall_color="green")

        # 模拟预测（全1.0向量）
        mock_pred_vector = np.ones(16)
        pred_walls = reconstruct_walls(calibrated_room_poly, mock_pred_vector, gt_masks)
        plot_room_analysis(
            calibrated_room_poly, mock_pred_vector, gt_masks, ax2, pred_walls, wall_color="red"
        )

        # 计算 IoU
        iou = calculate_wall_iou(gt_walls, pred_walls, buffer_width=0.5)
        iou_list.append(iou)

    print(f"平均 IoU: {np.mean(iou_list):.4f}")
    plt.show()
