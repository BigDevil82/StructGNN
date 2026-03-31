import glob
import os
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from preprocess.dxf_extractor import DXFExtractor
from preprocess.room_analyzer import get_room_edges
from preprocess.room_calibrator import RoomCalibrator

# ================= 数据结构定义 =================


@dataclass
class Node:
    """图节点：连接构件的物理点"""

    id: int
    x: float
    y: float
    # 记录该节点连接了哪些类型的边，用于后续GNN特征
    connected_types: Set[str] = field(default_factory=set)

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Segment:
    """边上的线段单元（构件）"""

    u: int  # Start Node ID
    v: int  # End Node ID
    type: str  # 'shear_wall', 'infill_wall', 'door', 'window', 'empty' (洞口), 'beam'

    # 物理属性
    length: float = 0.0

    # 所属房间索引（如果是梁则为None）
    room_index: Optional[int] = None
    # 在房间中的边缘索引 (0:上, 1:右, 2:下, 3:左)
    edge_index: int = -1


@dataclass
class RoomStruct:
    """房间结构化对象"""

    index: int
    polygon: Polygon
    # 房间的四条边，每条边是一个Segment列表
    # 顺序：Top, Right, Bottom, Left
    edges: List[List[Segment]] = field(default_factory=lambda: [[], [], [], []])


# ================= 核心处理类 =================


class NodeManager:
    """全局节点管理器，确保坐标相同的点具有唯一的ID"""

    def __init__(self, tol: float = 100.0):  # 100mm tolerance
        self.nodes: List[Node] = []
        self.pos_map: Dict[Tuple[int, int], int] = {}  # discretized pos -> id
        self.tol = tol

    def get_or_create(self, x: float, y: float) -> int:
        # 使用 grid snapping 避免浮点误差
        size = 5
        key = (int(round(x / size) * size), int(round(y / size) * size))

        if key in self.pos_map:
            return self.pos_map[key]

        node_id = len(self.nodes)
        self.nodes.append(Node(id=node_id, x=x, y=y))
        self.pos_map[key] = node_id
        return node_id

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]


class RoomEdgeProcessor:
    """处理单条房间边的构件映射与分割（包含自动缝合逻辑）"""

    def __init__(self, node_manager: NodeManager, tol: float = 10.0, gap_tol: float = 50.0):
        self.nm = node_manager
        self.tol = tol  # 几何投影的最小容差
        self.gap_tol = gap_tol  # 缝隙缝合阈值（小于此值的空隙会被填补）

    def project_and_resolve(
        self, base_edge: LineString, polygons_map: Dict[str, List[Polygon]]
    ) -> List[Dict]:

        edge_len = base_edge.length
        if edge_len < 1.0:
            return []

        # 1. 收集投影区间 (同前，略微优化)
        intervals = []
        priority_map = {"shear_wall": 10, "window": 5, "door": 5, "infill_wall": 1}

        for p_type, polys in polygons_map.items():
            if not polys:
                continue
            merged_poly = unary_union(polys)
            intersection = base_edge.intersection(merged_poly)

            if intersection.is_empty:
                continue

            intersect_geoms = []
            if isinstance(intersection, LineString):
                intersect_geoms.append(intersection)
            elif isinstance(intersection, MultiLineString):
                intersect_geoms.extend(intersection.geoms)
            elif hasattr(intersection, "geoms"):
                intersect_geoms.extend([g for g in intersection.geoms if isinstance(g, LineString)])

            for seg in intersect_geoms:
                if seg.length < self.tol:
                    continue
                d1 = base_edge.project(Point(seg.coords[0]))
                d2 = base_edge.project(Point(seg.coords[-1]))
                intervals.append(
                    {
                        "start": min(d1, d2),
                        "end": max(d1, d2),
                        "type": p_type,
                        "prio": priority_map.get(p_type, 0),
                    }
                )

        # 2. 扫描线分割 (Raw Segments)
        points = set([0.0, edge_len])
        for inv in intervals:
            points.add(inv["start"])
            points.add(inv["end"])

        sorted_points = sorted(list(points))
        raw_segments = []

        for i in range(len(sorted_points) - 1):
            s, e = sorted_points[i], sorted_points[i + 1]
            if e - s < 1.0:
                continue  # 忽略极小浮点误差

            mid = (s + e) / 2
            best_type = "empty"
            max_prio = -1

            for inv in intervals:
                if inv["start"] <= mid <= inv["end"]:
                    if inv["prio"] > max_prio:
                        max_prio = inv["prio"]
                        best_type = inv["type"]

            # 初步合并相同类型
            if raw_segments and raw_segments[-1]["type"] == best_type:
                raw_segments[-1]["end"] = e
            else:
                raw_segments.append({"start": s, "end": e, "type": best_type})

        # 3. 【关键步骤】缝隙清洗与缝合
        final_segments = self._clean_and_stitch(raw_segments, edge_len)

        return final_segments

    def _clean_and_stitch(self, segments: List[Dict], total_length: float) -> List[Dict]:
        """
        清洗分段数据：
        1. 消除微小的 'empty' 段（误差缝隙）。
        2. 缝合相邻构件。
        """
        if not segments:
            return []

        cleaned = []

        # --- 阶段一：过滤微小噪点 ---
        for seg in segments:
            length = seg["end"] - seg["start"]

            # 如果是极短的 empty 段 (小于容差)，大概率是绘图误差
            # 标记为 None，稍后处理（或者直接丢弃让后续逻辑填补）
            if seg["type"] == "empty" and length < self.gap_tol:
                continue

            # 如果是极短的有实体构件，通常保留，或者根据需求决定是否过滤
            # 这里假设有实体的构件都是有意义的
            cleaned.append(seg)

        if not cleaned:
            # 如果全被过滤了（比如全是极小的缝隙），直接返回一条完整的 empty
            return [{"start": 0.0, "end": total_length, "type": "empty"}]

        # --- 阶段二：缝合空隙 (Gap Stitching) ---
        # 经过阶段一，segments 之间可能出现了断层（因为删除了小片段）
        # 我们需要拉伸相邻片段来填补这些断层

        stitched = []

        # 处理首个片段：如果起点不是0且差距很小，吸附到0
        if cleaned[0]["start"] > 0 and cleaned[0]["start"] < self.gap_tol:
            cleaned[0]["start"] = 0.0

        stitched.append(cleaned[0])

        for i in range(1, len(cleaned)):
            prev = stitched[-1]
            curr = cleaned[i]

            gap = curr["start"] - prev["end"]

            # 情况A: 存在微小缝隙 (0 < gap < gap_tol)
            # 策略: "平分秋色" 或 "向后延伸"。这里采用平分，更公平
            if 0 < gap < self.gap_tol:
                mid = (prev["end"] + curr["start"]) / 2
                prev["end"] = mid
                curr["start"] = mid

            # 情况B: 存在大缝隙 (gap >= gap_tol)
            # 说明这是真的洞口，或者被阶段一删除的 empty 段确实很长
            elif gap >= self.gap_tol:
                # 显式插入一个 empty 段
                stitched.append({"start": prev["end"], "end": curr["start"], "type": "empty"})

            # 情况C: 重叠 (gap < 0) - 理论上扫描线算法不会产生，但防个万一
            elif gap < 0:
                mid = (prev["end"] + curr["start"]) / 2
                prev["end"] = mid
                curr["start"] = mid

            # 合并同类项（缝合后可能两个剪力墙连在一起了）
            if stitched[-1]["type"] == curr["type"]:
                stitched[-1]["end"] = curr["end"]
            else:
                stitched.append(curr)

        # 处理末尾：如果终点没到 total_length 且差距很小，吸附
        if stitched[-1]["end"] < total_length and (total_length - stitched[-1]["end"]) < self.gap_tol:
            stitched[-1]["end"] = total_length
        # 如果末尾确实缺一段，补上 empty
        elif stitched[-1]["end"] < total_length:
            stitched.append({"start": stitched[-1]["end"], "end": total_length, "type": "empty"})

        return stitched


class StructuralGraphBuilder:
    def __init__(self, dxf_path: str):
        self.extractor = DXFExtractor()
        self.dxf_data = self.extractor.extract_from_file(dxf_path)

        self.node_manager = NodeManager()
        self.edge_processor = RoomEdgeProcessor(self.node_manager)

        self.rooms: List[RoomStruct] = []
        self.beams: List[Segment] = []

        self.calibrated_rooms_poly: List[Polygon] = []

    def process(self):
        # print("1. 校准房间坐标...")
        self._calibrate_rooms()

        # print("2. 映射构件到房间边...")
        self._map_components_to_rooms()

        # print("3. 处理梁并吸附到节点...")
        self._process_beams()

        # print("4. 全局节点校准与对齐...")
        self._calibrate_nodes()

        print(
            f"处理完成: {len(self.rooms)} 个房间, {len(self.beams)} 条梁, {len(self.node_manager.nodes)} 个节点"
        )

    def _calibrate_rooms(self):
        self.raw_polys = [Polygon([(p.x, p.y) for p in r.polygon]) for r in self.extractor.rooms]
        self.calibrated_rooms_poly = RoomCalibrator(alignment_threshold=200.0).calibrate_rooms(self.raw_polys)

    def _map_components_to_rooms(self):
        # 准备构件字典
        components = {
            "shear_wall": [Polygon([(p.x, p.y) for p in w]) for w in self.extractor.shear_walls],
            "infill_wall": [Polygon([(p.x, p.y) for p in w]) for w in self.extractor.infill_walls],
            "door": [Polygon([(p.x, p.y) for p in w]) for w in self.extractor.doors],
            "window": [Polygon([(p.x, p.y) for p in w]) for w in self.extractor.windows],
        }

        for i, room_poly in enumerate(self.calibrated_rooms_poly):
            room_struct = RoomStruct(index=i, polygon=room_poly)

            # 获取房间的四条边 (Top, Right, Bottom, Left)
            # get_room_edges 返回的是 LineString 列表
            edges = get_room_edges(room_poly)

            for edge_idx, edge_geom in enumerate(edges):
                # 核心逻辑：投影并分割
                segments_info = self.edge_processor.project_and_resolve(edge_geom, components)

                current_edge_segments = []
                for info in segments_info:
                    # 计算物理坐标
                    p_start = edge_geom.interpolate(info["start"])
                    p_end = edge_geom.interpolate(info["end"])

                    # 获取/创建全局节点
                    u_id = self.node_manager.get_or_create(p_start.x, p_start.y)
                    v_id = self.node_manager.get_or_create(p_end.x, p_end.y)

                    # 更新节点属性
                    self.node_manager.get_node(u_id).connected_types.add(info["type"])
                    self.node_manager.get_node(v_id).connected_types.add(info["type"])

                    # 创建构件 Segment
                    seg = Segment(
                        u=u_id,
                        v=v_id,
                        type=info["type"],
                        length=info["end"] - info["start"],
                        room_index=i,
                        edge_index=edge_idx,
                    )
                    current_edge_segments.append(seg)

                room_struct.edges[edge_idx] = current_edge_segments

            self.rooms.append(room_struct)

    # def _process_beams(self):
    #     """将梁吸附到最近的节点（Node）上"""
    #     # 建立 KD-Tree 或者简单的距离查找
    #     # 这里为了演示，使用简单的最近点遍历（数据量不大）
    #     # 更好的方式是利用 grid 索引

    #     grid_tol = 250.0  # 梁吸附容差

    #     existing_nodes = self.node_manager.nodes
    #     seen_beams: Set[Tuple[int, int]] = set()

    #     for beam in self.extractor.beams:
    #         p1 = beam.start_point
    #         p2 = beam.end_point

    #         # 查找最近的节点
    #         u_id = self._find_nearest_node(p1.x, p1.y, existing_nodes, grid_tol)
    #         v_id = self._find_nearest_node(p2.x, p2.y, existing_nodes, grid_tol)

    #         if u_id is None or v_id is None:
    #             beam_type = "isolated_beam"
    #             # continue  # NOTE: 暂时跳过无法吸附的梁, 后面再想办法处理
    #         else:
    #             beam_type = "beam"

    #         # 如果找不到现有点，说明梁悬空或者连接到了非墙体交接处
    #         # 这种情况下，可能需要创建新节点（Beam End），或者这本身是一个数据错误
    #         # 这里策略：如果没有近点，创建新节点
    #         if u_id is None:
    #             u_id = self.node_manager.get_or_create(p1.x, p1.y)
    #         if v_id is None:
    #             v_id = self.node_manager.get_or_create(p2.x, p2.y)

    #         if u_id == v_id:
    #             continue  # 忽略长度为0的梁

    #         beam_key = tuple(sorted((u_id, v_id)))
    #         if beam_key in seen_beams:
    #             continue  # 跳过重复梁
    #         seen_beams.add(beam_key)

    #         # 创建梁 Segment
    #         n_u = self.node_manager.get_node(u_id)
    #         n_v = self.node_manager.get_node(v_id)

    #         self.beams.append(
    #             Segment(
    #                 u=u_id,
    #                 v=v_id,
    #                 type=beam_type,
    #                 length=np.hypot(n_u.x - n_v.x, n_u.y - n_v.y),
    #                 room_index=None,
    #             )
    #         )

    def _process_beams(self):
        """
        [修改版] 生成梁的 Ground Truth
        规则：房间边上，除了剪力墙以外的所有部分，都应该布置梁。
        这意味着：填充墙、门、窗、以及空洞（Empty）的位置，都需要生成梁数据。
        """
        self.beams = []  # 清空可能存在的旧数据
        seen_beams: Set[Tuple[int, int]] = set()

        # 遍历所有已处理好的房间
        for room in self.rooms:
            # 遍历房间的4条边 (Top, Right, Bottom, Left)
            for edge_segments in room.edges:
                # 遍历该边上的所有分段
                for seg in edge_segments:
                    # 核心规则：非剪力墙段 -> 必须布梁
                    if seg.type != "shear_wall":

                        # 1. 基础检查
                        if seg.u == seg.v:
                            continue  # 忽略长度为0的段

                        # 2. 去重逻辑 (无向边)
                        # 确保 u < v，作为唯一键
                        u, v = sorted((seg.u, seg.v))
                        beam_key = (u, v)

                        if beam_key in seen_beams:
                            continue  # 该位置已经添加过梁了（通常是相邻房间共享的边）

                        seen_beams.add(beam_key)

                        # 3. 添加到梁列表
                        # 注意：这里我们创建一个新的 Segment 对象作为“梁”
                        # 类型标记为 'beam'，用于后续训练作为正样本
                        new_beam = Segment(
                            u=u,
                            v=v,
                            type="beam",  # 标记为梁
                            length=seg.length,
                            room_index=None,  # 梁属于全局结构，不特定属于某个房间（虽然源自房间）
                            edge_index=-1,
                        )
                        self.beams.append(new_beam)

        # print(f"已根据剪力墙互补规则生成 {len(self.beams)} 条 Ground Truth 梁")

    def _calibrate_nodes(self):
        """
        最后一步：对所有节点进行全局坐标聚类校准。
        这能确保所有构件真正“水平”或“竖直”，消除微小倾斜。
        """
        if not self.node_manager.nodes:
            return

        # 1. 提取所有坐标
        xs = [n.x for n in self.node_manager.nodes]
        ys = [n.y for n in self.node_manager.nodes]

        # 2. 使用 RoomCalibrator 的聚类算法计算“对齐网格”
        calibrator = RoomCalibrator(alignment_threshold=200.0)
        x_grid = calibrator._cluster_1d(xs)
        y_grid = calibrator._cluster_1d(ys)

        # 3. 将每个节点更新到最近的网格线上
        # 注意：这里我们直接修改 Node 对象的 x, y 属性
        # 因为 Python 对象是引用的，所以 RoomStruct 和 Beam 里的几何信息虽然存的是旧值（如果存了 LineString），
        # 但拓扑关系（Node ID）引用的是这个 Node 对象。
        # 如果需要更新 Segment.geometry 以便画图正确，需要重新生成 geometry，或者画图时直接取 Node 坐标。

        for node in self.node_manager.nodes:
            new_x = calibrator._snap_to_grid(node.x, x_grid)
            new_y = calibrator._snap_to_grid(node.y, y_grid)
            node.x = new_x
            node.y = new_y

    def _find_nearest_node(self, x, y, nodes, tol):
        best_id = None
        min_dist = float("inf")
        for node in nodes:
            dist = np.hypot(node.x - x, node.y - y)
            if dist < min_dist:
                min_dist = dist
                best_id = node.id

        if min_dist < tol:
            return best_id
        return None

    def visualize(self, save_path: Optional[str] = None):
        """可视化提取结果"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 颜色映射
        color_map = {
            "shear_wall": "red",
            "infill_wall": "gray",
            "window": "green",
            "door": "blue",
            "empty": "lightgreen",  # 虚线或浅色
            "beam": "orange",
            "isolated_beam": "purple",
        }

        # 1. 绘制房间背景
        import matplotlib.colors as mcolors

        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, r in enumerate(self.rooms):
            x, y = r.polygon.exterior.xy
            ax.fill(x, y, color=colors[i % len(colors)], alpha=0.1)
            # ax.plot(x, y, color='blue', lw=0.5, alpha=0.3)

        # 2. 绘制房间边上的构件
        for r in self.rooms:
            for edge_segs in r.edges:
                for seg in edge_segs:
                    if seg.type != "shear_wall":
                        continue
                    n_u = self.node_manager.get_node(seg.u)
                    n_v = self.node_manager.get_node(seg.v)
                    c = color_map.get(seg.type, "black")
                    lw = 2 if seg.type == "empty" else 4
                    ls = "--" if seg.type == "empty" else "-"
                    ax.plot(
                        [n_u.x, n_v.x],
                        [n_u.y, n_v.y],
                        color=c,
                        lw=lw,
                        linestyle=ls,
                        alpha=0.8,
                        label=seg.type,
                        zorder=2,
                        # solid_capstyle="butt",
                    )

        # 3. 绘制梁
        for beam in self.beams:
            n_u = self.node_manager.get_node(beam.u)
            n_v = self.node_manager.get_node(beam.v)
            ax.plot(
                [n_u.x, n_v.x],
                [n_u.y, n_v.y],
                color=color_map.get(beam.type, "black"),
                lw=4,
                label=beam.type,
                zorder=1,
                # solid_capstyle="butt",
            )

        # # 4. 绘制节点
        # x_nodes = [n.x for n in self.node_manager.nodes]
        # y_nodes = [n.y for n in self.node_manager.nodes]
        # ax.scatter(x_nodes, y_nodes, c="green", s=5, zorder=10)

        # 去重图例
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(), loc="upper right")

        ax.set_aspect("equal")
        ax.set_title("结构化图纸数据提取 (GNN Pre-processing)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()


def process_single(dxf_path: str, output_image_path: Optional[str] = None):
    """单个文件处理函数（用于多进程）"""
    try:
        builder = StructuralGraphBuilder(dxf_path)
        builder.process()
        builder.visualize(save_path=output_image_path)
        return True, dxf_path
    except Exception as e:
        return False, dxf_path, str(e)


# ================= 使用入口 =================
if __name__ == "__main__":

    # 输入目录
    input_dir = r"data/dxf/to_process/beam_finish_modified_with_rooms"
    # 输出目录
    output_dir = r"data/dxf/plots/all_beam_pred"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有dxf文件
    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))
    total_files = len(dxf_files)
    print(f"找到 {total_files} 个DXF文件待处理。")

    # 准备任务列表
    tasks = []
    for dxf_file in dxf_files:
        filename = os.path.basename(dxf_file)
        file_stem = os.path.splitext(filename)[0]
        # if file_stem != "L1L28_25":
        #     continue  # 仅处理指定文件，调试用
        save_path = os.path.join(output_dir, f"{file_stem}.png")
        tasks.append((dxf_file, save_path))

    print(f"准备处理 {len(tasks)} 个文件，使用 {cpu_count()} 个进程。")

    # 多进程处理
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(process_single, tasks)

    # 统计结果
    success_count = sum(1 for r in results if r[0])
    print(f"\n处理完成: {success_count}/{len(tasks)} 个文件成功")

    # 打印失败的文件
    failed = [r for r in results if not r[0]]
    if failed:
        print("\n失败的文件:")
        for _, dxf_path, error in failed:
            print(f"  {os.path.basename(dxf_path)}: {error}")
