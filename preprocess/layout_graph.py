"""
布局图构建模块

功能：
1. 根据校准后的房间布局构建图结构（NetworkX DiGraph）
2. 提取节点特征：几何信息（Box, Area）+ 剪力墙分布（Label）
3. 提取边特征：邻接关系、重合长度、相对方位（Top/Bottom/Left/Right）
4. 可视化：在平面图上叠加显示图拓扑结构

图结构说明：
- 节点：每个房间对应一个节点
- 边：相邻房间（共享边）之间建立有向边
- 节点特征：归一化几何特征 + 剪力墙向量 + 约束掩码
- 边特征：共享边长度、位置、方向等
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

from dxf_extractor import DXFExtractor
from preprocess.room_analyzer import RoomAnalyzer, plot_room_analysis
from preprocess.room_calibrator import calibrate_rooms


class LayoutGraphBuilder:
    """
    布局图构建器

    将房间布局转换为图结构，用于GNN模型输入
    """

    def __init__(self, rooms: List[Polygon], room_indices: List[int] = None):
        """
        Args:
            rooms: 校准后的房间多边形列表
            room_indices: 房间的原始索引列表（用于追踪），默认自动生成 0..N-1
        """
        self.rooms = rooms
        self.num_rooms = len(rooms)
        self.room_indices = room_indices if room_indices else list(range(self.num_rooms))

        # 计算户型整体的包围盒和归一化参数
        self._compute_normalization_params()

        # 初始化有向图
        self.graph = nx.DiGraph()
        self._init_nodes()
        self._build_edges()

    def _compute_normalization_params(self):
        """
        计算户型整体的包围盒和归一化参数

        目的：将所有房间的坐标归一化到[-1, 1]范围，
             使模型对不同尺度的户型具有更好的泛化能力
        """
        all_bounds = [room.bounds for room in self.rooms]
        total_minx = min(b[0] for b in all_bounds)
        total_miny = min(b[1] for b in all_bounds)
        total_maxx = max(b[2] for b in all_bounds)
        total_maxy = max(b[3] for b in all_bounds)

        # 户型中心点
        self.total_center_x = (total_minx + total_maxx) / 2
        self.total_center_y = (total_miny + total_maxy) / 2

        # 归一化尺度因子（取最大边长）
        self.scale_factor = max(total_maxx - total_minx, total_maxy - total_miny)

        # 避免除零
        if self.scale_factor < 1e-6:
            self.scale_factor = 1.0

    def _init_nodes(self):
        """
        初始化图节点及基础几何特征

        每个节点包含：
        - poly: 房间多边形
        - geo_feature: 归一化几何特征向量(9维)
        - sw_vector: 剪力墙向量(16维) - 后续通过add_analysis_results添加
        - masks: 可布置区域掩码 - 后续添加
        """
        for i, room in enumerate(self.rooms):
            room_id = self.room_indices[i]
            minx, miny, maxx, maxy = room.bounds
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            width = maxx - minx
            height = maxy - miny
            area = width * height

            # 归一化到 [-1, 1] 范围
            norm_minx = (minx - self.total_center_x) / (self.scale_factor / 2)
            norm_miny = (miny - self.total_center_y) / (self.scale_factor / 2)
            norm_maxx = (maxx - self.total_center_x) / (self.scale_factor / 2)
            norm_maxy = (maxy - self.total_center_y) / (self.scale_factor / 2)
            norm_width = width / (self.scale_factor / 2)
            norm_height = height / (self.scale_factor / 2)
            norm_area = area / (self.scale_factor**2)
            norm_cx = (center_x - self.total_center_x) / (self.scale_factor / 2)
            norm_cy = (center_y - self.total_center_y) / (self.scale_factor / 2)

            # 归一化后的几何特征向量 (9维)
            # [minx, miny, maxx, maxy, width, height, area, center_x, center_y]
            geo_feature = np.array(
                [
                    norm_minx,
                    norm_miny,
                    norm_maxx,
                    norm_maxy,
                    norm_width,
                    norm_height,
                    norm_area,
                    norm_cx,
                    norm_cy,
                ],
                dtype=np.float32,
            )

            # 添加节点
            self.graph.add_node(
                room_id,
                poly=room,
                geo_feature=geo_feature,
                # 剪力墙标签将在后续通过 add_analysis_results 添加
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

                # 归一化边特征
                norm_shared_length = shared_length / (self.scale_factor / 2)
                norm_overlap_cx = (centroid.x - self.total_center_x) / (self.scale_factor / 2)
                norm_overlap_cy = (centroid.y - self.total_center_y) / (self.scale_factor / 2)

                # 边特征向量 (7维)
                # [norm_shared_length, norm_overlap_cx, norm_overlap_cy, is_top, is_bottom, is_left, is_right, is_horizontal]
                edge_feature = np.concatenate(
                    [[norm_shared_length, norm_overlap_cx, norm_overlap_cy], direction_vec, [is_horizontal]]
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
            ax.fill(x, y, color="#A8A8A8", alpha=0.15)

            # 绘制节点 (质心) - 使用原始坐标
            cx, cy = room.centroid.x, room.centroid.y
            ax.scatter(cx, cy, c="green", s=300, zorder=5, edgecolors="white")
            if show_labels:
                ax.text(
                    cx,
                    cy,
                    str(node_id),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    zorder=6,
                )

        # 2. 绘制边
        for u, v in self.graph.edges:
            # 使用原始坐标
            room_u = self.graph.nodes[u]["poly"]
            room_v = self.graph.nodes[v]["poly"]
            pos_u = [room_u.centroid.x, room_u.centroid.y]
            pos_v = [room_v.centroid.x, room_v.centroid.y]

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

            # 可选：绘制边的接触点/重叠中心 (使用原始几何)
            shared_geom = self.graph.edges[u, v]["shared_geom"]
            overlap_center = shared_geom.centroid
            ax.scatter(overlap_center.x, overlap_center.y, c="red", s=20, marker="x", zorder=5)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Room Graph Structure")


def convert_to_graph(dxf_path: str, save_path: str = None):

    # 1. 提取
    if not os.path.exists(dxf_path):
        # print(f"文件不存在: {dxf_path}")
        exit()

    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    # 转换几何体
    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    sw_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.shear_walls]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

    # 2. 校准
    # print(f"校准前房间数: {len(raw_rooms)}")
    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200)
    # print("房间坐标已校准")

    # 3. 分析 (获取 Ground Truth)
    # Note: 这一步用的是未校准的原始room, 因为校准后可能导致墙边不在构件多边形框内部
    # 获取分析结果后，原始room_poly就没用了，一切基于校准后的room_poly操作
    analyzer = RoomAnalyzer(sw_polys, infill_polys)
    analysis_results = []
    # print("正在分析房间剪力墙分布...")
    for i, room in enumerate(raw_rooms):
        if not room.is_valid or room.area < 1:
            raise ValueError(f"房间{i}无效或面积小于1")
        sw_vector, masks = analyzer.process_room(room)
        analysis_results.append({"room_index": i, "sw_vector": sw_vector.tolist(), "masks": masks})

    # 4. 构建图
    # print("构建图结构...")
    graph_builder = LayoutGraphBuilder(calibrated_rooms)
    graph_builder.add_analysis_results(analysis_results)

    # 5. 可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    # 绘制图结构
    graph_builder.visualize(ax, show_labels=True)

    # 叠加剪力墙 Ground Truth (可选，为了验证对齐情况)
    # 这里只画一部分验证
    for res in analysis_results:
        room = calibrated_rooms[res["room_index"]]
        plot_room_analysis(room, np.array(res["sw_vector"]), res["masks"], ax, wall_color="green")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # print(f"图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


# ==========================================
# 整合测试 pipeline
# ==========================================
if __name__ == "__main__":
    # test single file
    dxf_path = r"dxf/to_process/room_finished/L1L28_232.dxf"
    convert_to_graph(dxf_path)

    # # convert batch files
    # from tqdm import tqdm

    # dxf_dir = r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\dxf\to_process\room_finished"
    # dxf_files = [f for f in os.listdir(dxf_dir) if f.endswith(".dxf")]
    # for dxf_file in tqdm(dxf_files):
    #     dxf_path = os.path.join(dxf_dir, dxf_file)
    #     save_path = Path(dxf_path).with_suffix(".png")
    #     convert_to_graph(dxf_path, save_path)
