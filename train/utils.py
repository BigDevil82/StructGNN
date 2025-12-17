"""
工具函数模块
提取重复使用的函数，避免代码冗余
"""

from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon

from train.augmentor import GeometryAugmentor
from train.config import data_config


def compute_anchor_ratios(intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
    """计算边两端的剪力墙比例"""
    if not intervals:
        return 0.0, 0.0
    if len(intervals) == 1:
        if intervals[0][0] < 1e-3:
            return intervals[0][1], 0.0
        else:
            return 0.0, 1.0 - intervals[0][0]
    start_ratio = intervals[0][1] if intervals[0][0] < 1e-3 else 0.0
    end_ratio = 1.0 - intervals[-1][0]
    return start_ratio, end_ratio


def mask_to_constraint_vector(
    buildable_masks: List[List[Tuple[float, float]]], threshold: float = None
) -> np.ndarray:
    """
    将可布置区域掩码列表转换为约束向量

    Args:
        buildable_masks: 8个半边的可布置区间列表，每个元素为 [(start, end), ...]
        threshold: 可布置比例阈值，低于此值的端点标记为不可布置

    Returns:
        constraint_vector: 16维向量 (1.0=可布置, 0.0=不可布置)
    """
    if threshold is None:
        threshold = data_config.BUILDABLE_THRESHOLD

    # 初始化为全1（假设所有位置都可布置）
    constraint_vector = np.ones(16, dtype=np.float32)

    # 对每个半边，计算起始和结束端点的可布置比例
    buildable_ratios = []
    for intervals in buildable_masks:
        start_ratio, end_ratio = compute_anchor_ratios(intervals)
        buildable_ratios.extend([start_ratio, end_ratio])

    # 根据阈值判定是否可布置
    for i, ratio in enumerate(buildable_ratios):
        if ratio < threshold:
            constraint_vector[i] = 0.0

    return constraint_vector


def extract_dxf_geometry(dxf_path: str) -> Tuple[List[Polygon], List[Polygon], List[Polygon]]:
    """
    从DXF文件提取几何数据

    Args:
        dxf_path: DXF文件路径

    Returns:
        (raw_rooms, sw_polys, infill_polys): 房间、剪力墙、填充墙的多边形列表
    """
    from preprocess.dxf_extractor import DXFExtractor

    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    # 转换为Shapely几何体
    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

    return raw_rooms, sw_polys, infill_polys


def build_graph_from_dxf(dxf_path: str, mode: str = "none"):
    """
    从DXF文件构建完整的图数据结构

    流程：提取 → 增广 → 校准 → 分析 → 构图

    Args:
        dxf_path: DXF文件路径
        mode: 数据增广模式，可选值: "none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270"

    Returns:
        (graph_builder, calibrated_rooms, analysis_results):
            - graph_builder: LayoutGraphBuilder实例
            - calibrated_rooms: 校准后的房间列表
            - analysis_results: 剪力墙分析结果列表
    """
    from preprocess.layout_graph import LayoutGraphBuilder
    from preprocess.room_analyzer import RoomAnalyzer
    from preprocess.room_calibrator import calibrate_rooms

    # 1. 提取几何数据
    raw_rooms, sw_polys, infill_polys = extract_dxf_geometry(dxf_path)

    if not raw_rooms:
        raise ValueError(f"DXF文件中没有有效房间: {dxf_path}")

    raw_rooms, sw_polys, infill_polys = GeometryAugmentor.apply_augmentation(
        raw_rooms, sw_polys, infill_polys, mode
    )

    # 2. 校准房间坐标
    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=data_config.ALIGNMENT_THRESHOLD)

    # 3. 分析剪力墙分布（使用原始房间匹配墙体）
    analyzer = RoomAnalyzer(sw_polys, infill_polys)
    analysis_results = []

    for i, room in enumerate(raw_rooms):
        if not room.is_valid or room.area < 1:
            continue

        sw_vector, masks = analyzer.process_room(room)
        analysis_results.append({"room_index": i, "sw_vector": sw_vector, "masks": masks})

    # 4. 构建图结构
    graph_builder = LayoutGraphBuilder(calibrated_rooms)
    graph_builder.add_analysis_results(analysis_results)

    return graph_builder, calibrated_rooms, analysis_results
