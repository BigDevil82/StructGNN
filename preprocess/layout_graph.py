"""
布局图构建模块 (Graph Construction)

功能：
1. 根据校准后的房间布局构建图结构 (NetworkX DiGraph)
2. 提取节点特征：几何信息 (Box, Area) + 剪力墙分布 (Label)
3. 提取边特征：邻接关系、重合长度、相对方位 (Top/Bottom/Left/Right)
4. 可视化：在平面图上叠加显示图拓扑结构
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon


class LayoutGraphBuilder:
    """布局图构建器"""

    def __init__(self, rooms: List[Polygon], room_indices: List[int] = None):
        """
        Args:
            rooms: 校准后的房间多边形列表
            room_indices: 房间的原始索引列表（用于追踪），默认自动生成 0..N
        """
        self.rooms = rooms
        self.num_rooms = len(rooms)
        self.room_indices = room_indices if room_indices else list(range(self.num_rooms))

        # 初始化有向图
        self.graph = nx.DiGraph()
        self._init_nodes()
        self._build_edges()

    def _init_nodes(self):
        """初始化图节点及基础几何特征"""
        for i, room in enumerate(self.rooms):
            room_id = self.room_indices[i]
            minx, miny, maxx, maxy = room.bounds
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            width = maxx - minx
            height = maxy - miny
            area = width * height

            # 基础几何特征向量 (9维)
            # [minx, miny, maxx, maxy, width, height, area, cx, cy]
            geo_feature = np.array(
                [minx, miny, maxx, maxy, width, height, area, center_x, center_y], dtype=np.float32
            )

            self.graph.add_node(
                room_id,
                poly=room,
                geo_feature=geo_feature,
                # 后续可通过 add_node_features 添加更多信息（如 shear wall label）
                sw_vector=None,
                masks=None,
            )

    def _get_relative_position(self, src_bounds, tgt_bounds) -> np.ndarray:
        """
        判断 Target 相对于 Source 的方位
        Returns:
            One-hot vector: [Top, Bottom, Left, Right]
        """
        # 使用中心点判断相对方位
        src_cx = (src_bounds[0] + src_bounds[2]) / 2
        src_cy = (src_bounds[1] + src_bounds[3]) / 2
        tgt_cx = (tgt_bounds[0] + tgt_bounds[2]) / 2
        tgt_cy = (tgt_bounds[1] + tgt_bounds[3]) / 2

        dx = tgt_cx - src_cx
        dy = tgt_cy - src_cy

        # 简单的主要方向判断 (基于矩形几何)
        # 由于是校准后的曼哈顿几何，这种判断通常足够准确
        # [Top, Bottom, Left, Right]
        direction = np.zeros(4, dtype=np.float32)

        if abs(dy) > abs(dx):
            if dy > 0:
                direction[0] = 1.0  # Top
            else:
                direction[1] = 1.0  # Bottom
        else:
            if dx < 0:
                direction[2] = 1.0  # Left
            else:
                direction[3] = 1.0  # Right

        return direction

    def _build_edges(self):
        """
        构建边：检测邻接关系并计算边特征
        规则：如果两个房间相交或接触（共享边），则建立双向边
        """
        for i in range(self.num_rooms):
            for j in range(self.num_rooms):
                if i == j:
                    continue

                room_i = self.rooms[i]
                room_j = self.rooms[j]
                id_i = self.room_indices[i]
                id_j = self.room_indices[j]

                # 检查相交或接触
                if not room_i.intersects(room_j) and not room_i.touches(room_j):
                    continue

                # 计算重合部分（几何交集）
                intersection = room_i.intersection(room_j)

                # 只有当交集非空（且通常是线段或矩形）时才认为有效连接
                # 忽略点接触 (Point)
                if intersection.is_empty or isinstance(intersection, Point):
                    continue

                # 计算重合长度
                shared_length = intersection.length
                if shared_length < 400:
                    continue

                # 如果是重叠（Polygon），则长度取周长的一半作为近似或取0
                if isinstance(intersection, Polygon):
                    shared_length = intersection.length / 2

                # 计算重合中心点
                centroid = intersection.centroid

                # 计算相对方位
                direction_vec = self._get_relative_position(room_i.bounds, room_j.bounds)

                # 水平/垂直边标志
                is_horizontal = direction_vec[0] == 1.0 or direction_vec[1] == 1.0

                # 边特征向量 (7维)
                # [shared_length, overlap_cx, overlap_cy, is_top, is_bottom, is_left, is_right, is_horizontal]
                edge_feature = np.concatenate(
                    [[shared_length, centroid.x, centroid.y], direction_vec, [is_horizontal]]
                ).astype(np.float32)

                self.graph.add_edge(id_i, id_j, feature=edge_feature, shared_geom=intersection)

    def add_analysis_results(self, analysis_results: List[dict]):
        """
        将 RoomAnalyzer 的分析结果（剪力墙向量等）添加到图节点中作为 Label 或 Feature
        """
        result_map = {res["room_index"]: res for res in analysis_results}

        for node_id in self.graph.nodes:
            # 这里假设 room_indices 与 analysis_result 中的 room_index 一致
            # 如果 LayoutGraphBuilder 初始化时传入了特定的 indices，需要注意对应关系
            if node_id in result_map:
                res = result_map[node_id]
                self.graph.nodes[node_id]["sw_vector"] = np.array(res["sw_vector"], dtype=np.float32)
                self.graph.nodes[node_id]["masks"] = res["masks"]

    def get_pyg_data(self):
        """
        (预留接口) 将图转换为 PyTorch Geometric Data 对象
        Returns:
            data object compatible with PyG
        """
        # 这里仅作伪代码展示，避免引入 torch 依赖导致当前环境报错
        # import torch
        # from torch_geometric.data import Data
        # x = torch.tensor([self.graph.nodes[i]['geo_feature'] for i in ...])
        # edge_index = ...
        # edge_attr = ...
        # y = torch.tensor([self.graph.nodes[i]['sw_vector'] for i in ...])
        # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        pass

    def visualize(self, ax: plt.Axes, show_labels: bool = True):
        """
        在给定的 Axes 上可视化房间和图结构
        """
        # 1. 绘制房间底图 (复用 RoomAnalyzer 风格)
        for node_id in self.graph.nodes:
            room: Polygon = self.graph.nodes[node_id]["poly"]
            x, y = room.exterior.xy
            ax.plot(x, y, color="black", linewidth=2, alpha=0.5)
            ax.fill(x, y, color="lightgray", alpha=0.4)

            # 绘制节点 (质心)
            geo = self.graph.nodes[node_id]["geo_feature"]
            cx, cy = geo[7], geo[8]  # center_x, center_y
            ax.scatter(cx, cy, c="green", s=400, zorder=5, edgecolors="white")
            if show_labels:
                ax.text(
                    cx,
                    cy,
                    str(node_id),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    zorder=6,
                )

        # 2. 绘制边
        for u, v in self.graph.edges:
            pos_u = self.graph.nodes[u]["geo_feature"][7:9]  # [cx, cy]
            pos_v = self.graph.nodes[v]["geo_feature"][7:9]

            # 绘制连接线
            ax.plot(
                [pos_u[0], pos_v[0]],
                [pos_u[1], pos_v[1]],
                color="orange",
                linestyle="-",
                linewidth=3.5,
                alpha=0.6,
                zorder=4,
            )

            # 可选：绘制边的接触点/重叠中心
            overlap_center = self.graph.edges[u, v]["feature"][1:3]
            ax.scatter(overlap_center[0], overlap_center[1], c="red", s=20, marker="x", zorder=5)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Room Graph Structure")


# ==========================================
# 整合测试 pipeline
# ==========================================
if __name__ == "__main__":
    import os
    import sys

    # 确保能导入同级模块
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from dxf_extractor import DXFExtractor
    from preprocess.room_analyzer import RoomAnalyzer, plot_room_analysis
    from preprocess.room_calibrator import calibrate_rooms

    print("=" * 60)
    print("Pipeline 测试：DXF -> 校准 -> 分析 -> 图构建")
    print("=" * 60)

    # 1. 提取
    dxf_path = r"dxf/to_process/room_finished/L1L28_232.dxf"
    if not os.path.exists(dxf_path):
        print(f"文件不存在: {dxf_path}")
        exit()

    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    # 转换几何体
    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

    # 2. 校准
    print(f"校准前房间数: {len(raw_rooms)}")
    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)
    print("房间坐标已校准")

    # 3. 分析 (获取 Ground Truth)
    # Note: 这一步用的是未校准的原始room, 因为校准后可能导致墙边不在构件多边形框内部
    # 获取分析结果后，原始room_poly就没用了，一切基于校准后的room_poly操作
    analyzer = RoomAnalyzer(sw_polys, infill_polys)
    analysis_results = []

    print("正在分析房间剪力墙分布...")
    for i, room in enumerate(raw_rooms):
        if not room.is_valid or room.area < 1:
            raise ValueError(f"房间{i}无效或面积小于1")
        sw_vector, masks = analyzer.process_room(room)
        analysis_results.append({"room_index": i, "sw_vector": sw_vector.tolist(), "masks": masks})

    # 4. 构建图
    print("构建图结构...")
    graph_builder = LayoutGraphBuilder(calibrated_rooms)
    graph_builder.add_analysis_results(analysis_results)

    G = graph_builder.graph
    print(f"图构建完成:")
    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")

    # 打印一个节点的示例数据
    if len(G.nodes) > 0:
        node_0 = G.nodes[0]
        print("\n示例节点 (ID=0) 特征:")
        print(f"  - Geo Feature (shape={node_0['geo_feature'].shape}): {node_0['geo_feature'][:4]}...")
        if node_0["sw_vector"] is not None:
            print(f"  - SW Vector (shape={node_0['sw_vector'].shape})")

    # 5. 可视化
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制图结构
    graph_builder.visualize(ax, show_labels=True)

    # 叠加剪力墙 Ground Truth (可选，为了验证对齐情况)
    # 这里只画一部分验证
    for res in analysis_results:
        room = calibrated_rooms[res["room_index"]]
        plot_room_analysis(room, np.array(res["sw_vector"]), res["masks"], ax, wall_color="green")

    plt.show()
