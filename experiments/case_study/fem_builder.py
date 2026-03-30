import json
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon
from shapely.ops import snap, unary_union

from preprocess.room_analyzer import _walls_from_vector, get_room_edges


class FEMTopologyBuilder:
    """
    基于线段运算的有限元拓扑构建器 v2.2

    核心改进：
    1. 用线段intersection/difference代替buffer，避免正交处间隙
    2. 按边分类后用区间减法精确计算梁位置
    3. snap对齐确保精确连接
    """

    def __init__(self, snap_tolerance: float = 2.0, gap_tolerance: float = 5.0, min_length: float = 10.0):
        """
        Args:
            snap_tolerance: 坐标对齐容差
            gap_tolerance: 区间合并容差（合并相邻的剪力墙段）
            min_length: 最小构件长度
        """
        self.snap_tolerance = snap_tolerance
        self.gap_tolerance = gap_tolerance
        self.min_length = min_length

        self.raw_walls: List[LineString] = []
        self.room_edges: List[LineString] = []
        self.room_polys: List[Polygon] = []

        self.nodes: List[Tuple[float, float]] = []
        self.elements: List[dict] = []

    def add_room(self, room_poly: Polygon, sw_vector: np.ndarray, masks: List = None):
        """收集房间几何信息"""
        self.room_polys.append(room_poly)
        self.room_edges.extend(get_room_edges(room_poly))
        walls_geom = _walls_from_vector(room_poly, sw_vector, masks)
        if not walls_geom.is_empty:
            if isinstance(walls_geom, LineString):
                self.raw_walls.append(walls_geom)
            elif hasattr(walls_geom, "geoms"):
                self.raw_walls.extend(list(walls_geom.geoms))
        else:
            print("  [FEMBuilder] 警告: 房间没有检测到剪力墙")

    def build(self) -> dict:
        print("  [FEMBuilder] 正在构建拓扑 (Interval-based)...")

        # 1. 合并边界线和剪力墙线
        skeleton_lines = self._merge_collinear_lines(self.room_edges)
        wall_lines = self._merge_collinear_lines(self.raw_walls) if self.raw_walls else []

        print(f"    边界线: {len(self.room_edges)} -> {len(skeleton_lines)} (merged)")
        print(f"    剪力墙: {len(self.raw_walls)} -> {len(wall_lines)} (merged)")

        # 2. 对每条边界线，用区间减法计算梁
        segments = []

        for skel_line in skeleton_lines:
            # 找出与此边界线共线的剪力墙
            overlapping_walls = self._find_overlapping_segments(skel_line, wall_lines)

            if not overlapping_walls:
                # 整条边都是梁
                segments.append({"geometry": skel_line, "type": "beam", "length": skel_line.length})
            else:
                # 用区间减法计算梁和剪力墙
                wall_segs, beam_segs = self._subtract_wall_from_edge(skel_line, overlapping_walls)
                segments.extend(wall_segs)
                segments.extend(beam_segs)

        # 3. 过滤短构件
        # segments = [s for s in segments if s["length"] >= self.min_length]

        # 4. 在交叉点处打断构件，确保梁-梁、梁-墙、墙-墙相交处都有节点
        segments = self._split_at_intersections(segments)

        # 5. 再次过滤打断后的短构件
        # segments = [s for s in segments if s["length"] >= self.min_length]

        # 6. 构建节点拓扑
        self._build_topology(segments)

        return self._format_result()

    def _merge_collinear_lines(self, lines: List[LineString]) -> List[LineString]:
        """合并共线线段"""
        if not lines:
            return []

        # 分类：水平、垂直、其他
        horizontal: Dict[int, List[Tuple[int, int]]] = {}
        vertical: Dict[int, List[Tuple[int, int]]] = {}
        others: List[LineString] = []

        tolerance = 2

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            p1, p2 = coords[0], coords[-1]
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            x2, y2 = int(round(p2[0])), int(round(p2[1]))

            if abs(y1 - y2) < tolerance:
                y_key = y1
                x_min, x_max = min(x1, x2), max(x1, x2)
                if y_key not in horizontal:
                    horizontal[y_key] = []
                horizontal[y_key].append((x_min, x_max))
            elif abs(x1 - x2) < tolerance:
                x_key = x1
                y_min, y_max = min(y1, y2), max(y1, y2)
                if x_key not in vertical:
                    vertical[x_key] = []
                vertical[x_key].append((y_min, y_max))
            else:
                others.append(line)

        merged_lines = []

        # 合并水平线段
        for y, intervals in horizontal.items():
            merged = self._merge_intervals(intervals)
            for x_min, x_max in merged:
                if x_max - x_min >= 1:
                    merged_lines.append(LineString([(x_min, y), (x_max, y)]))

        # 合并垂直线段
        for x, intervals in vertical.items():
            merged = self._merge_intervals(intervals)
            for y_min, y_max in merged:
                if y_max - y_min >= 1:
                    merged_lines.append(LineString([(x, y_min), (x, y_max)]))

        merged_lines.extend(others)
        return merged_lines

    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠或相邻的区间"""
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(sorted_intervals[0])]

        for start, end in sorted_intervals[1:]:
            if start <= merged[-1][1] + self.gap_tolerance:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return [(m[0], m[1]) for m in merged]

    def _find_overlapping_segments(
        self, skel_line: LineString, wall_lines: List[LineString]
    ) -> List[Tuple[float, float]]:
        """
        找出与边界线共线且重叠的剪力墙区间

        Returns:
            区间列表，每个区间是 (start_ratio, end_ratio) 在 [0, 1] 范围内
        """
        skel_coords = list(skel_line.coords)
        if len(skel_coords) < 2:
            return []

        p1, p2 = np.array(skel_coords[0]), np.array(skel_coords[-1])
        skel_vec = p2 - p1
        skel_len = np.linalg.norm(skel_vec)

        if skel_len < 1:
            return []

        skel_dir = skel_vec / skel_len

        # 判断是水平还是垂直
        is_horizontal = abs(skel_dir[1]) < 0.1
        is_vertical = abs(skel_dir[0]) < 0.1

        if not (is_horizontal or is_vertical):
            return []  # 斜线暂不处理

        overlapping = []
        tolerance = self.snap_tolerance

        for wall in wall_lines:
            wall_coords = list(wall.coords)
            if len(wall_coords) < 2:
                continue

            w1, w2 = np.array(wall_coords[0]), np.array(wall_coords[-1])

            # 检查是否共线
            if is_horizontal:
                # 水平线：检查y坐标是否相同
                if abs(w1[1] - p1[1]) > tolerance or abs(w2[1] - p1[1]) > tolerance:
                    continue
                # 计算x方向的重叠
                skel_x_min, skel_x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
                wall_x_min, wall_x_max = min(w1[0], w2[0]), max(w1[0], w2[0])

                # 计算交集
                inter_min = max(skel_x_min, wall_x_min)
                inter_max = min(skel_x_max, wall_x_max)

                if inter_max > inter_min + 1:
                    # 转换为比例
                    ratio_min = (inter_min - skel_x_min) / (skel_x_max - skel_x_min)
                    ratio_max = (inter_max - skel_x_min) / (skel_x_max - skel_x_min)
                    overlapping.append((ratio_min, ratio_max))

            elif is_vertical:
                # 垂直线：检查x坐标是否相同
                if abs(w1[0] - p1[0]) > tolerance or abs(w2[0] - p1[0]) > tolerance:
                    continue
                # 计算y方向的重叠
                skel_y_min, skel_y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
                wall_y_min, wall_y_max = min(w1[1], w2[1]), max(w1[1], w2[1])

                inter_min = max(skel_y_min, wall_y_min)
                inter_max = min(skel_y_max, wall_y_max)

                if inter_max > inter_min + 1:
                    ratio_min = (inter_min - skel_y_min) / (skel_y_max - skel_y_min)
                    ratio_max = (inter_max - skel_y_min) / (skel_y_max - skel_y_min)
                    overlapping.append((ratio_min, ratio_max))

        return overlapping

    def _subtract_wall_from_edge(
        self, skel_line: LineString, wall_intervals: List[Tuple[float, float]]
    ) -> Tuple[List[dict], List[dict]]:
        """
        从边界线中减去剪力墙区间，得到剪力墙和梁

        Args:
            skel_line: 边界线
            wall_intervals: 剪力墙区间列表 [(ratio_min, ratio_max), ...]

        Returns:
            (wall_segments, beam_segments)
        """
        # 合并重叠的剪力墙区间
        if not wall_intervals:
            return [], [{"geometry": skel_line, "type": "beam", "length": skel_line.length}]

        sorted_intervals = sorted(wall_intervals, key=lambda x: x[0])
        merged = [[sorted_intervals[0][0], sorted_intervals[0][1]]]

        for start, end in sorted_intervals[1:]:
            # 使用较小的容差合并相邻区间
            if start <= merged[-1][1] + 0.02:  # 2% 容差
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        wall_intervals = [(m[0], m[1]) for m in merged]

        # 计算梁区间：[0, 1] - wall_intervals
        beam_intervals = []
        current = 0.0

        for w_start, w_end in wall_intervals:
            if w_start > current + 0.01:
                beam_intervals.append((current, w_start))
            current = max(current, w_end)

        if current < 0.99:
            beam_intervals.append((current, 1.0))

        # 转换为线段
        coords = list(skel_line.coords)
        p1, p2 = np.array(coords[0]), np.array(coords[-1])
        vec = p2 - p1

        wall_segments = []
        beam_segments = []

        for w_start, w_end in wall_intervals:
            start_pt = tuple(p1 + vec * w_start)
            end_pt = tuple(p1 + vec * w_end)
            line = LineString([start_pt, end_pt])
            wall_segments.append({"geometry": line, "type": "shearwall", "length": line.length})

        for b_start, b_end in beam_intervals:
            start_pt = tuple(p1 + vec * b_start)
            end_pt = tuple(p1 + vec * b_end)
            line = LineString([start_pt, end_pt])
            if line.length >= 400:
                beam_segments.append({"geometry": line, "type": "beam", "length": line.length})
            else:
                # 过短的梁可能是误差，合并到剪力墙
                wall_segments.append({"geometry": line, "type": "shearwall", "length": line.length})

        return wall_segments, beam_segments

    def _extract_lines(self, geom) -> List[LineString]:
        """从混合几何体中提取 LineString"""
        lines = []
        if geom.is_empty:
            return lines

        if isinstance(geom, LineString):
            lines.append(geom)
        elif isinstance(geom, (MultiLineString, GeometryCollection)):
            for g in geom.geoms:
                if isinstance(g, LineString):
                    lines.append(g)
                elif isinstance(g, MultiLineString):
                    lines.extend(g.geoms)
        return lines

    def _split_at_intersections(self, segments: List[dict]) -> List[dict]:
        """
        在所有构件的交叉点处打断，确保相交处产生共享节点。

        处理两类情况：
        1. 两条线段相交（十字交叉、T字交叉）→ 在交点处打断
        2. 一条线段的端点落在另一条线段内部 → 在该端点处打断

        只处理水平/垂直线段的正交相交。
        """
        tolerance = self.snap_tolerance

        # 收集所有需要在每条线段上插入的打断点
        # key: segment index, value: list of ratio values (0~1) along the segment
        split_ratios: Dict[int, List[float]] = {i: [] for i in range(len(segments))}

        for i in range(len(segments)):
            line_i = segments[i]["geometry"]
            ci = list(line_i.coords)
            pi1, pi2 = np.array(ci[0]), np.array(ci[-1])
            vec_i = pi2 - pi1
            len_i = np.linalg.norm(vec_i)
            if len_i < 1:
                continue

            # 判断水平/垂直
            is_h_i = abs(vec_i[1]) < tolerance
            is_v_i = abs(vec_i[0]) < tolerance
            if not (is_h_i or is_v_i):
                continue

            for j in range(i + 1, len(segments)):
                line_j = segments[j]["geometry"]
                cj = list(line_j.coords)
                pj1, pj2 = np.array(cj[0]), np.array(cj[-1])
                vec_j = pj2 - pj1
                len_j = np.linalg.norm(vec_j)
                if len_j < 1:
                    continue

                is_h_j = abs(vec_j[1]) < tolerance
                is_v_j = abs(vec_j[0]) < tolerance
                if not (is_h_j or is_v_j):
                    continue

                # 只处理正交情况（水平vs垂直）
                if is_h_i == is_h_j:
                    # 共线方向：端点可能落在对方内部
                    self._check_collinear_splits(
                        pi1, pi2, len_i, pj1, pj2, len_j, is_h_i, tolerance, split_ratios, i, j
                    )
                else:
                    # 正交交叉
                    self._check_orthogonal_intersection(
                        pi1, pi2, vec_i, len_i, is_h_i, pj1, pj2, vec_j, len_j, tolerance, split_ratios, i, j
                    )

        # 根据打断点拆分线段
        result = []
        for idx, seg in enumerate(segments):
            ratios = split_ratios[idx]
            if not ratios:
                result.append(seg)
                continue

            # 去重并排序
            ratios = sorted(set(ratios))
            # 去掉过于接近端点的打断点
            ratios = [r for r in ratios if 0.01 < r < 0.99]
            if not ratios:
                result.append(seg)
                continue

            # 用打断点拆分
            coords = list(seg["geometry"].coords)
            p1, p2 = np.array(coords[0]), np.array(coords[-1])
            vec = p2 - p1

            breakpoints = [0.0] + ratios + [1.0]
            for k in range(len(breakpoints) - 1):
                r_start = breakpoints[k]
                r_end = breakpoints[k + 1]
                start_pt = tuple(p1 + vec * r_start)
                end_pt = tuple(p1 + vec * r_end)
                new_line = LineString([start_pt, end_pt])
                if new_line.length >= 1:
                    result.append(
                        {
                            "geometry": new_line,
                            "type": seg["type"],
                            "length": new_line.length,
                        }
                    )

        n_splits = len(result) - len(segments)
        if n_splits > 0:
            print(f"    交叉打断: {len(segments)} -> {len(result)} 段 (+{n_splits})")

        return result

    def _check_collinear_splits(
        self, pi1, pi2, len_i, pj1, pj2, len_j, is_horizontal, tolerance, split_ratios, i, j
    ):
        """检查共线线段的端点是否落在对方内部，若是则添加打断点"""
        if is_horizontal:
            # 水平线段：检查y是否相同
            if abs(pi1[1] - pj1[1]) > tolerance:
                return
            # i的范围
            xi_min, xi_max = min(pi1[0], pi2[0]), max(pi1[0], pi2[0])
            xj_min, xj_max = min(pj1[0], pj2[0]), max(pj1[0], pj2[0])

            # j的端点落在i内部
            for xp in [pj1[0], pj2[0]]:
                if xi_min + tolerance < xp < xi_max - tolerance:
                    r = abs(xp - pi1[0]) / len_i
                    split_ratios[i].append(r)
            # i的端点落在j内部
            for xp in [pi1[0], pi2[0]]:
                if xj_min + tolerance < xp < xj_max - tolerance:
                    r = abs(xp - pj1[0]) / len_j
                    split_ratios[j].append(r)
        else:
            # 垂直线段
            if abs(pi1[0] - pj1[0]) > tolerance:
                return
            yi_min, yi_max = min(pi1[1], pi2[1]), max(pi1[1], pi2[1])
            yj_min, yj_max = min(pj1[1], pj2[1]), max(pj1[1], pj2[1])

            for yp in [pj1[1], pj2[1]]:
                if yi_min + tolerance < yp < yi_max - tolerance:
                    r = abs(yp - pi1[1]) / len_i
                    split_ratios[i].append(r)
            for yp in [pi1[1], pi2[1]]:
                if yj_min + tolerance < yp < yj_max - tolerance:
                    r = abs(yp - pj1[1]) / len_j
                    split_ratios[j].append(r)

    def _check_orthogonal_intersection(
        self, pi1, pi2, vec_i, len_i, is_h_i, pj1, pj2, vec_j, len_j, tolerance, split_ratios, i, j
    ):
        """检查正交线段是否相交，若是则在交点处添加打断点"""
        if is_h_i:
            # i 水平, j 垂直
            h_y = pi1[1]
            h_xmin, h_xmax = min(pi1[0], pi2[0]), max(pi1[0], pi2[0])
            v_x = pj1[0]
            v_ymin, v_ymax = min(pj1[1], pj2[1]), max(pj1[1], pj2[1])
        else:
            # i 垂直, j 水平
            h_y = pj1[1]
            h_xmin, h_xmax = min(pj1[0], pj2[0]), max(pj1[0], pj2[0])
            v_x = pi1[0]
            v_ymin, v_ymax = min(pi1[1], pi2[1]), max(pi1[1], pi2[1])

        # 检查交点是否在两条线段的范围内（不含端点附近）
        if not (
            h_xmin + tolerance < v_x < h_xmax - tolerance and v_ymin + tolerance < h_y < v_ymax - tolerance
        ):
            # 交点不在两条线段的严格内部 → 可能是T字或端点连接
            # 仍需检查：交点是否至少在一条线段内部（T字情况）
            in_h = h_xmin - tolerance <= v_x <= h_xmax + tolerance
            in_v = v_ymin - tolerance <= h_y <= v_ymax + tolerance
            if not (in_h and in_v):
                return

        # 交点坐标
        cross_x, cross_y = v_x, h_y

        # 计算交点在 i 上的ratio
        if len_i > 1:
            ri = np.dot(np.array([cross_x, cross_y]) - pi1, vec_i) / (len_i * len_i)
            if 0.01 < ri < 0.99:
                split_ratios[i].append(ri)

        # 计算交点在 j 上的ratio
        if len_j > 1:
            rj = np.dot(np.array([cross_x, cross_y]) - pj1, vec_j) / (len_j * len_j)
            if 0.01 < rj < 0.99:
                split_ratios[j].append(rj)

    def _build_topology(self, segments: List[dict]):
        """构建节点和构件列表"""
        self.nodes = []
        self.elements = []
        node_map = {}

        def get_node_id(pt):
            # 坐标取整
            key = (round(pt[0]), round(pt[1]))

            if key not in node_map:
                node_map[key] = len(self.nodes)
                self.nodes.append(key)
            return node_map[key]

        for seg in segments:
            line = seg["geometry"]
            coords = list(line.coords)

            u = get_node_id(coords[0])
            v = get_node_id(coords[-1])

            if u == v:
                continue

            self.elements.append(
                {
                    "id": len(self.elements),
                    "start_node": u,
                    "end_node": v,
                    "start_coord": self.nodes[u],
                    "end_coord": self.nodes[v],
                    "type": seg["type"],
                    "length": seg["length"],
                }
            )

    def validate_topology(self, result: Optional[dict] = None) -> dict:
        """
        校核拓扑中是否存在孤立点与异常构件。

        检查内容：
        1. 孤立点: 在节点列表中，但没有任何构件连接的点。
        2. 悬挑构件: 仅一端节点度为 1 的构件（另一端连接到主结构）。
        3. 非主连通分量: 不在最大连通分量中的节点和构件（常见于漂浮子结构）。

        Args:
            result: 可选，build() 返回的结果。若不提供则使用当前实例的 self.nodes/self.elements。

        Returns:
            拓扑校核报告字典。
        """
        if result is None:
            nodes = self.nodes
            members = self.elements
        else:
            nodes = result.get("nodes", [])
            members = result.get("members", [])

        num_nodes = len(nodes)
        degree = [0 for _ in range(num_nodes)]

        adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
        member_ids_in_graph: Set[int] = set()

        for member in members:
            u = member["start_node"]
            v = member["end_node"]

            if 0 <= u < num_nodes:
                degree[u] += 1
            if 0 <= v < num_nodes:
                degree[v] += 1

            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adjacency[u].add(v)
                adjacency[v].add(u)
                member_ids_in_graph.add(member["id"])

        isolated_node_ids = [i for i, d in enumerate(degree) if d == 0]

        dangling_members = []
        for member in members:
            u = member["start_node"]
            v = member["end_node"]

            du = degree[u] if 0 <= u < num_nodes else 0
            dv = degree[v] if 0 <= v < num_nodes else 0

            # 仅一端悬挑（异或）：恰好一个端点度 <= 1
            if (du <= 1) != (dv <= 1):
                dangling_members.append(
                    {
                        "id": member["id"],
                        "start_node": u,
                        "end_node": v,
                        "start_degree": du,
                        "end_degree": dv,
                        "type": member.get("type"),
                        "length": member.get("length"),
                    }
                )

        visited = set()
        components = []

        for node_id in range(num_nodes):
            if node_id in visited:
                continue
            if degree[node_id] == 0:
                visited.add(node_id)
                continue

            stack = [node_id]
            comp_nodes = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp_nodes.add(cur)
                for nxt in adjacency[cur]:
                    if nxt not in visited:
                        stack.append(nxt)

            if comp_nodes:
                components.append(comp_nodes)

        largest_comp_nodes: Set[int] = max(components, key=len) if components else set()
        floating_node_ids = sorted(
            [n for n in range(num_nodes) if degree[n] > 0 and n not in largest_comp_nodes]
        )

        floating_member_ids = []
        for member in members:
            if member["start_node"] in floating_node_ids and member["end_node"] in floating_node_ids:
                floating_member_ids.append(member["id"])

        has_issue = bool(isolated_node_ids or dangling_members or floating_member_ids)

        report = {
            "ok": not has_issue,
            "summary": {
                "num_nodes": num_nodes,
                "num_members": len(members),
                "num_isolated_nodes": len(isolated_node_ids),
                "num_dangling_members": len(dangling_members),
                "num_connected_components": len(components),
                "num_floating_nodes": len(floating_node_ids),
                "num_floating_members": len(floating_member_ids),
            },
            "isolated_nodes": [{"id": i, "coord": nodes[i]} for i in isolated_node_ids],
            "dangling_members": dangling_members,
            "floating_components": {
                "node_ids": floating_node_ids,
                "member_ids": floating_member_ids,
            },
        }

        if report["ok"]:
            print("    拓扑校核通过: 未发现孤立点/悬挑构件/漂浮子结构")
        else:
            print(
                "    拓扑校核警告: "
                f"孤立点={report['summary']['num_isolated_nodes']}, "
                f"悬挑构件={report['summary']['num_dangling_members']}, "
                f"漂浮构件={report['summary']['num_floating_members']}"
            )

        return report

    def _compute_slabs(self) -> List[List[Tuple]]:
        """
        计算楼板区域，覆盖所有房间及其间的空洞。

        做法：
        1. 合并所有 room_polys 得到整体轮廓（可能含 holes）
        2. 对每个连通区域：将外轮廓作为一块楼板
        3. 对该区域内部的每个 hole（空洞）：也作为一块楼板（填满过道/大厅等）

        Returns:
            楼板坐标列表，每个楼板为角点 (x, y) 列表（首尾不重复）
        """
        if not self.room_polys:
            return []

        merged = unary_union(self.room_polys)

        # 统一处理为列表
        if isinstance(merged, Polygon):
            polys = [merged]
        else:
            polys = list(merged.geoms)

        slabs = []
        for poly in polys:
            if poly.is_empty:
                continue
            # # 外轮廓（无洞）作为一整块楼板
            # slabs.append(list(poly.exterior.coords)[:-1])
            # 每个空洞单独作为一块楼板
            for interior in poly.interiors:
                slabs.append(list(interior.coords)[:-1])
        for room in self.room_polys:
            slabs.append(list(room.exterior.coords)[:-1])

        # sum area for statistics
        total_slab_area = sum(Polygon(slab).area for slab in slabs)
        print(f"    计算楼板: {len(slabs)} 块, 总面积约 {total_slab_area:.2f} 平方单位")

        return slabs

    def _format_result(self) -> dict:
        slabs = self._compute_slabs()
        validation = self.validate_topology()
        return {
            "nodes": self.nodes,
            "members": self.elements,
            "slabs": slabs,
            "validation": validation,
            "statistics": {
                "num_nodes": len(self.nodes),
                "min_xy": np.min(self.nodes, axis=0).tolist(),
                "max_xy": np.max(self.nodes, axis=0).tolist(),
                "num_members": len(self.elements),
                "num_shearwalls": sum(1 for m in self.elements if m["type"] == "shearwall"),
                "num_beams": sum(1 for m in self.elements if m["type"] == "beam"),
                "num_slabs": len(slabs),
                "total_sw_length": sum(m["length"] for m in self.elements if m["type"] == "shearwall"),
                "total_beam_length": sum(m["length"] for m in self.elements if m["type"] == "beam"),
            },
        }


def visualize_fem_result(
    result: dict,
    room_polys: List[Polygon] = None,
    title: str = "FEM Members",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    可视化FEM解析结果

    Args:
        result: EdgeBasedFEMParser.build() 的返回结果
        room_polys: 房间多边形列表（用于绘制背景）
        title: 图标题
        save_path: 保存路径
        figsize: 图尺寸
    """
    _, ax = plt.subplots(figsize=figsize)

    nodes = result["nodes"]
    members = result["members"]
    validation = result.get("validation", {})

    dangling_member_map = {m["id"]: m for m in validation.get("dangling_members", [])}
    dangling_member_ids = set(dangling_member_map.keys())
    floating_member_ids = set(validation.get("floating_components", {}).get("member_ids", []))

    # 绘制房间轮廓（背景）
    if room_polys:
        for room in room_polys:
            x, y = room.exterior.xy
            ax.fill(x, y, alpha=0.1, color="lightblue", edgecolor="gray", linestyle="--", linewidth=0.5)

    # 绘制构件
    for member in members:
        start = member["start_coord"]
        end = member["end_coord"]

        if member["type"] == "shearwall":
            color = "red"
            linewidth = 4
            zorder = 10
        else:  # beam
            color = "blue"
            linewidth = 4
            zorder = 2

        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, zorder=zorder)

    # 叠加绘制异常构件高亮
    for member in members:
        mid = member.get("id")
        if mid in floating_member_ids:
            start = member["start_coord"]
            end = member["end_coord"]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="orange",
                linewidth=5,
                linestyle="--",
                alpha=0.9,
                zorder=30,
            )

    for member in members:
        mid = member.get("id")
        if mid in dangling_member_ids:
            start = member["start_coord"]
            end = member["end_coord"]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="magenta",
                linewidth=5,
                linestyle=":",
                alpha=0.95,
                zorder=35,
            )

            # 标记悬挑自由端
            info = dangling_member_map[mid]
            if info["start_degree"] <= 1 and info["end_degree"] > 1:
                free_pt = start
            elif info["end_degree"] <= 1 and info["start_degree"] > 1:
                free_pt = end
            else:
                free_pt = None

            if free_pt is not None:
                ax.scatter([free_pt[0]], [free_pt[1]], color="magenta", marker="x", s=70, zorder=40)

    # 绘制节点
    node_x = [n[0] for n in nodes]
    node_y = [n[1] for n in nodes]
    ax.scatter(node_x, node_y, color="black", s=20, zorder=20)

    # # 添加节点编号（可选）
    # if len(nodes) <= 100:
    #     for i, (x, y) in enumerate(nodes):
    #         ax.annotate(str(i), (x, y), fontsize=6, ha="center", va="bottom")

    # 图例
    legend_elements = [
        Line2D(
            [0], [0], color="red", linewidth=4, label=f'Shear Wall ({result["statistics"]["num_shearwalls"]})'
        ),
        Line2D([0], [0], color="blue", linewidth=2, label=f'Beam ({result["statistics"]["num_beams"]})'),
        Line2D(
            [0],
            [0],
            color="magenta",
            linewidth=4,
            linestyle=":",
            label=f"Cantilever ({len(dangling_member_ids)})",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            linewidth=4,
            linestyle="--",
            label=f"Floating ({len(floating_member_ids)})",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     marker="o",
        #     color="w",
        #     markerfacecolor="black",
        #     markersize=6,
        #     label=f'Nodes ({result["statistics"]["num_nodes"]})',
        # ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def export_to_json(result: dict, output_path: str):
    """
    导出结果到JSON文件（用于ETABS建模）

    Args:
        result: EdgeBasedFEMParser.build() 的返回结果
        output_path: 输出文件路径
    """
    export_data = {
        "nodes": [{"id": i, "x": n[0], "y": n[1]} for i, n in enumerate(result["nodes"])],
        "shearwalls": [
            {
                "start": list(m["start_coord"]),
                "end": list(m["end_coord"]),
                "start_node": m["start_node"],
                "end_node": m["end_node"],
                "length": m["length"],
            }
            for m in result["members"]
            if m["type"] == "shearwall"
        ],
        "beams": [
            {
                "start": list(m["start_coord"]),
                "end": list(m["end_coord"]),
                "start_node": m["start_node"],
                "end_node": m["end_node"],
                "length": m["length"],
            }
            for m in result["members"]
            if m["type"] == "beam"
        ],
        "slabs": result["slabs"],
        "statistics": result["statistics"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Exported FEM data to: {output_path}")
