import bisect
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from preprocess.dxf_extractor import DXFExtractor
from preprocess.room_analyzer import get_room_edges
from preprocess.room_calibrator import RoomCalibrator, calibrate_rooms


@dataclass
class Node:
    """图节点对象"""

    id: int
    x: float
    y: float
    type: str  # 'corner' (房间角点), 'wall_end' (剪力墙端点), 'beam_node' (梁连接点)

    # 可选：记录来源信息，方便回溯
    source_ref: Optional[str] = None

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Edge:
    """图边对象"""

    u: int  # Start Node ID
    v: int  # End Node ID
    type: str  # 'wall' (剪力墙), 'beam' (梁), 'room_boundary' (房间虚线)

    # 物理属性
    length: float = 0.0
    is_horizontal: bool = False

    # 原始几何对象 (方便可视化或后续精细处理)
    geometry: Optional[object] = None

    def __post_init__(self):
        # 自动保证无向图的 u < v，方便去重
        if self.u > self.v:
            self.u, self.v = self.v, self.u

    def key(self) -> Tuple[int, int]:
        return (self.u, self.v)


# ================= 核心处理类 =================


class StructuralGraphBuilder:
    def __init__(self, dxf_path: str):
        self.extractor = DXFExtractor()
        self.dxf_data = self.extractor.extract_from_file(dxf_path)

        # 核心数据容器
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        # 辅助查找表
        self.node_map: Dict[Tuple[float, float], int] = {}  # (x, y) -> node_id
        self.edge_set: Set[Tuple[int, int]] = set()  # (u, v) -> exist?

        # 轴向索引 (用于梁吸附)
        self.nodes_on_x_axis = defaultdict(list)  # x_coord -> [node_id, ...]
        self.nodes_on_y_axis = defaultdict(list)  # y_coord -> [node_id, ...]

        # 全局轴网
        self.x_grid: List[float] = []
        self.y_grid: List[float] = []
        self.raw_rooms: List[Polygon] = []
        self.calibrated_rooms: List[Polygon] = []

        # 参数配置 (mm)
        self.GRID_TOL = 200.0  # 吸附到轴网的容差
        self.NODE_TOL = 150.0  # 节点合并容差
        self.EDGE_LEN_MIN = 300.0  # 忽略过短的边

    # ---------- 基础工具方法 ----------

    def _snap_val(self, val: float, grid: List[float], tol: float) -> Tuple[float, bool]:
        """二分查找快速吸附数值"""
        if not grid:
            return val, False
        idx = bisect.bisect_left(grid, val)
        candidates = []
        if idx < len(grid):
            candidates.append(grid[idx])
        if idx > 0:
            candidates.append(grid[idx - 1])

        if not candidates:
            return val, False
        nearest = min(candidates, key=lambda x: abs(x - val))

        if abs(nearest - val) <= tol:
            return nearest, True
        return val, False

    def get_or_create_node(self, x: float, y: float, node_type: str) -> int:
        """获取现有节点或创建新节点，并更新所有索引"""
        # 使用 round 避免浮点精度问题导致的重复创建
        key = (round(x, 1), round(y, 1))

        if key in self.node_map:
            # 如果节点已存在，尝试更新类型优先级 (例如 wall_end > corner)
            existing_id = self.node_map[key]
            # 简单的优先级逻辑：如果是 'wall_end'，覆盖 'corner'
            if node_type == "wall_end" and self.nodes[existing_id].type == "corner":
                self.nodes[existing_id].type = "wall_end"
            return existing_id

        # 创建新节点
        new_id = len(self.nodes)
        node = Node(id=new_id, x=x, y=y, type=node_type)
        self.nodes.append(node)
        self.node_map[key] = new_id

        # 更新轴向索引
        self.nodes_on_x_axis[round(x, 1)].append(new_id)
        self.nodes_on_y_axis[round(y, 1)].append(new_id)

        return new_id

    def add_edge(self, u: int, v: int, edge_type: str, geometry=None):
        """添加边（自动去重）"""
        if u == v:
            return

        # 构造临时 Edge 对象以获取规范化的 (u, v) key
        temp_edge = Edge(u, v, edge_type)
        key = temp_edge.key()

        if key in self.edge_set:
            return

        # 计算物理属性
        p1, p2 = self.nodes[u], self.nodes[v]
        length = np.hypot(p1.x - p2.x, p1.y - p2.y)
        is_horz = abs(p1.x - p2.x) > abs(p1.y - p2.y)

        temp_edge.length = length
        temp_edge.is_horizontal = is_horz
        temp_edge.geometry = geometry

        self.edges.append(temp_edge)
        self.edge_set.add(key)

    def _find_nearest_node_on_axis(
        self, target_val: float, axis_coord: float, axis_dict: dict
    ) -> Optional[int]:
        """在指定轴线上查找最近节点"""
        axis_key = round(axis_coord, 1)
        if axis_key not in axis_dict:
            return None

        candidate_ids = axis_dict[axis_key]
        best_id = None
        min_dist = self.NODE_TOL

        for nid in candidate_ids:
            node = self.nodes[nid]
            # 判断是在比较 X 还是 Y
            # 如果查的是 nodes_on_y_axis (固定Y查X)，则比较 node.x
            curr_val = node.x if axis_dict is self.nodes_on_y_axis else node.y

            dist = abs(curr_val - target_val)
            if dist < min_dist:
                min_dist = dist
                best_id = nid
        return best_id

    # ---------- 核心流程拆解 ----------

    def _init_grid(self):
        """步骤1: 房间校准与轴网提取"""
        self.raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in self.extractor.rooms]

        calibrator = RoomCalibrator(alignment_threshold=200.0)
        self.calibrated_rooms = calibrator.calibrate_rooms(self.raw_rooms)

        self.x_grid = sorted(calibrator.x_grid)
        self.y_grid = sorted(calibrator.y_grid)

    def _process_walls_mapping(self):
        """步骤2: 墙体映射 (Raw -> Normalized -> Calibrated)"""
        sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in self.extractor.shear_walls]
        sw_union = unary_union(sw_polys)

        # 同时遍历原始房间和校准房间
        for raw_room, cal_room in zip(self.raw_rooms, self.calibrated_rooms):

            # 1. 先把校准后的房间角点入库
            cal_corners = list(cal_room.exterior.coords)[:4]
            for cx, cy in cal_corners:
                self.get_or_create_node(cx, cy, node_type="corner")

            # 2. 获取对应的边 (Raw 和 Calibrated 顺序必须一致: 上, 右, 下, 左)
            raw_edges = get_room_edges(raw_room)
            cal_edges = get_room_edges(cal_room)

            for raw_edge, cal_edge in zip(raw_edges, cal_edges):
                # 在原始空间中找重叠
                intersects = []
                intersection = raw_edge.intersection(sw_union)
                if intersection.is_empty:
                    continue

                if isinstance(intersection, LineString):
                    intersects.append(intersection)
                elif isinstance(intersection, MultiLineString):
                    intersects.extend(intersection.geoms)
                elif hasattr(intersection, "geoms"):
                    intersects.extend([g for g in intersection.geoms if isinstance(g, LineString)])

                # 将原始重叠段映射到校准边
                for seg in intersects:
                    if seg.length < self.EDGE_LEN_MIN:
                        continue  # 忽略过短噪点

                    # A. 计算原始端点在原始边上的归一化位置 (0.0 ~ 1.0)
                    # project 方法返回距离，除以长度得到比例
                    ratio_start = raw_edge.project(Point(seg.coords[0]), normalized=True)
                    ratio_end = raw_edge.project(Point(seg.coords[-1]), normalized=True)

                    # B. 在校准边上利用比例恢复坐标
                    p_start = cal_edge.interpolate(ratio_start, normalized=True)
                    p_end = cal_edge.interpolate(ratio_end, normalized=True)

                    # C. 创建节点和边 (使用校准后的坐标)
                    u = self.get_or_create_node(p_start.x, p_start.y, node_type="wall_end")
                    v = self.get_or_create_node(p_end.x, p_end.y, node_type="wall_end")

                    # 几何对象用于可视化，保存映射后的线段
                    mapped_geom = LineString([p_start, p_end])
                    self.add_edge(u, v, edge_type="wall", geometry=mapped_geom)

    def _process_beams_orthogonal(self):
        """步骤3: 梁的正交吸附 (两端必须吸附到竖向/横向轴网)"""
        for beam in self.extractor.beams:
            start, end = beam.start_point, beam.end_point
            x1, y1 = start.x, start.y
            x2, y2 = end.x, end.y

            dx, dy = abs(x1 - x2), abs(y1 - y2)
            is_horizontal = dx > dy

            if is_horizontal:
                # --- 水平梁 ---
                # 1. Y坐标: 吸附到水平轴线 (确保梁是平的)
                avg_y = (y1 + y2) / 2
                snapped_y, _ = self._snap_val(avg_y, self.y_grid, self.GRID_TOL)

                # 2. X坐标 (端点): 必须吸附到 **竖向轴线 (X Grid)**
                # 这样保证梁的端点落在某个网格交点上（即潜在的柱或墙位置）
                sx1, _ = self._snap_val(x1, self.x_grid, self.GRID_TOL)
                sx2, _ = self._snap_val(x2, self.x_grid, self.GRID_TOL)

                # 创建节点
                u = self.get_or_create_node(sx1, snapped_y, node_type="beam_node")
                v = self.get_or_create_node(sx2, snapped_y, node_type="beam_node")

            else:
                # --- 垂直梁 ---
                # 1. X坐标: 吸附到竖向轴线
                avg_x = (x1 + x2) / 2
                snapped_x, _ = self._snap_val(avg_x, self.x_grid, self.GRID_TOL)

                # 2. Y坐标 (端点): 必须吸附到 **水平轴线 (Y Grid)**
                sy1, _ = self._snap_val(y1, self.y_grid, self.GRID_TOL)
                sy2, _ = self._snap_val(y2, self.y_grid, self.GRID_TOL)
                # 创建节点
                u = self.get_or_create_node(snapped_x, sy1, node_type="beam_node")
                v = self.get_or_create_node(snapped_x, sy2, node_type="beam_node")

            self.add_edge(u, v, edge_type="beam")

    def process(self):
        self._init_grid()
        self._process_walls_mapping()
        self._process_beams_orthogonal()

    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 12))

        # 1. 绘制轴网 (最底层)
        for x in self.x_grid:
            ax.axvline(x, c="#e0e0e0", lw=0.5, zorder=0)
        for y in self.y_grid:
            ax.axhline(y, c="#e0e0e0", lw=0.5, zorder=0)

        # 2. 绘制房间 (填充透明色) [改进点]
        import matplotlib.colors as mcolors

        colors = list(mcolors.TABLEAU_COLORS.values())

        for i, room in enumerate(self.calibrated_rooms):
            x, y = room.exterior.xy
            c = colors[i % len(colors)]
            # 边框
            ax.plot(x, y, c="gray", ls="-", lw=0.5, alpha=0.6, zorder=1)
            # 填充
            ax.fill(x, y, c=c, alpha=0.1, zorder=1)

        # 3. 绘制边
        for edge in self.edges:
            p1 = self.nodes[edge.u]
            p2 = self.nodes[edge.v]

            if edge.type == "wall":
                ax.plot(
                    [p1.x, p2.x], [p1.y, p2.y], c="red", lw=4, solid_capstyle="butt", zorder=3, label="Wall"
                )
            elif edge.type == "beam":
                ax.plot([p1.x, p2.x], [p1.y, p2.y], c="orange", lw=4, zorder=4, label="Beam")

        # 4. 绘制节点
        for node in self.nodes:
            c = "black"
            s = 10
            if node.type == "corner":
                c = "gray"
                s = 5
            elif node.type == "beam_node":
                c = "blue"
                s = 20

            ax.scatter(node.x, node.y, c=c, s=s, zorder=5)

        # 去重图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")

        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()


# =================使用示例=================
if __name__ == "__main__":
    # 替换为你的 DXF 路径
    dxf_path = r"dxf/to_process/beam_finish_modified_with_rooms/L17_14.dxf"

    builder = StructuralGraphBuilder(dxf_path)
    builder.process()
    builder.visualize()

    # 获取结构化数据
    print(f"提取了 {len(builder.nodes)} 个节点")
    print(f"提取了 {len(builder.edges)} 条边")
