from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class RawMember:
    raw_id: int
    kind: str
    axis: str
    band_coord: float
    start: tuple[float, float]
    end: tuple[float, float]
    interval: tuple[float, float]
    start_node: int | None = None
    end_node: int | None = None
    length: float | None = None


@dataclass(frozen=True)
class MergedMember:
    node_id: int
    kind: str
    axis: str
    band_coord: float
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    raw_member_ids: tuple[int, ...]


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class MemberGraphBuilder:
    """
    将 FEM 构件归并为更粗粒度的图结构。

    规则：
    1. 按 墙 / 主梁 / 次梁 分组；
    2. 组内按 水平 / 竖直 分开；
    3. 在同一坐标带内，将连续共线段合并为一个图节点；
    4. 基于原始构件的连接关系，在合并节点之间建立无向边。
    """

    def __init__(self, coord_tol: float = 0.01, axis_tol: float | None = None):
        self.coord_tol = coord_tol
        self.axis_tol = coord_tol if axis_tol is None else axis_tol

    def build_graph(self, data: dict[str, Any]) -> nx.Graph:
        raw_members = self._load_members(data)
        graph = nx.Graph()
        if not raw_members:
            return graph

        merged_members, raw_to_node = self._merge_members(raw_members)
        for merged in merged_members:
            graph.add_node(
                merged.node_id,
                kind=merged.kind,
                axis=merged.axis,
                band_coord=merged.band_coord,
                start=merged.start,
                end=merged.end,
                length=merged.length,
                raw_member_ids=list(merged.raw_member_ids),
            )

        for (u, v), points in self._build_edges(raw_members, raw_to_node).items():
            graph.add_edge(u, v, intersection_points=[list(point) for point in sorted(points)])

        return graph

    def load_members(self, data: dict[str, Any]) -> list[RawMember]:
        return self._load_members(data)

    def build_graph_data(self, data: dict[str, Any]) -> dict[str, Any]:
        graph = self.build_graph(data)
        return {
            "nodes": [
                {
                    "id": node_id,
                    "kind": attrs["kind"],
                    "axis": attrs["axis"],
                    "band_coord": attrs["band_coord"],
                    "start": list(attrs["start"]),
                    "end": list(attrs["end"]),
                    "length": attrs["length"],
                    "raw_member_ids": attrs["raw_member_ids"],
                }
                for node_id, attrs in graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "intersection_points": attrs.get("intersection_points", []),
                }
                for u, v, attrs in graph.edges(data=True)
            ],
            "statistics": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_wall_nodes": sum(1 for _, a in graph.nodes(data=True) if a["kind"] == "wall"),
                "num_primary_beam_nodes": sum(
                    1 for _, a in graph.nodes(data=True) if a["kind"] == "primary_beam"
                ),
                "num_secondary_beam_nodes": sum(
                    1 for _, a in graph.nodes(data=True) if a["kind"] == "secondary_beam"
                ),
            },
        }

    def export_graph_json(self, data: dict[str, Any], output_path: str | Path):
        graph_data = self.build_graph_data(data)
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

    def _load_members(self, data: dict[str, Any]) -> list[RawMember]:
        if "members" in data:
            members = [self._from_result_member(m, idx) for idx, m in enumerate(data["members"])]
        else:
            members = []
            for idx, m in enumerate(data.get("shearwalls", [])):
                members.append(self._from_export_member(m, idx, "wall"))
            offset = len(members)
            for idx, m in enumerate(data.get("beams", [])):
                members.append(self._from_export_member(m, offset + idx, None))

        return [m for m in members if m is not None]

    def _from_result_member(self, member: dict[str, Any], raw_id: int) -> RawMember | None:
        kind = self._member_kind(member.get("type"), member.get("beam_role"))
        start = self._point(member["start_coord"])
        end = self._point(member["end_coord"])
        axis = self._axis_of(start, end)
        if axis is None:
            return None
        interval = self._interval_of(start, end, axis)
        return RawMember(
            raw_id=member.get("id", raw_id),
            kind=kind,
            axis=axis,
            band_coord=self._band_value(start, end, axis),
            start=start,
            end=end,
            interval=interval,
            start_node=member.get("start_node"),
            end_node=member.get("end_node"),
            length=float(member.get("length", self._distance(start, end))),
        )

    def _from_export_member(
        self, member: dict[str, Any], raw_id: int, preset_kind: str | None
    ) -> RawMember | None:
        kind = preset_kind or self._member_kind("beam", member.get("beam_role"))
        start = self._point(member["start"])
        end = self._point(member["end"])
        axis = self._axis_of(start, end)
        if axis is None:
            return None
        interval = self._interval_of(start, end, axis)
        return RawMember(
            raw_id=raw_id,
            kind=kind,
            axis=axis,
            band_coord=self._band_value(start, end, axis),
            start=start,
            end=end,
            interval=interval,
            start_node=member.get("start_node"),
            end_node=member.get("end_node"),
            length=float(member.get("length") or self._distance(start, end)),
        )

    def _merge_members(self, raw_members: list[RawMember]) -> tuple[list[MergedMember], dict[int, int]]:
        buckets: dict[tuple[str, str], list[RawMember]] = {}
        for member in raw_members:
            buckets.setdefault((member.kind, member.axis), []).append(member)

        merged_members: list[MergedMember] = []
        raw_to_node: dict[int, int] = {}

        for (kind, axis), members in buckets.items():
            for band in self._cluster_by_band(members):
                for components in self._connected_components(band):
                    merged = self._merge_component(components, len(merged_members), kind, axis)
                    merged_members.append(merged)
                    for member in components:
                        raw_to_node[member.raw_id] = merged.node_id

        return merged_members, raw_to_node

    def _cluster_by_band(self, members: list[RawMember]) -> list[list[RawMember]]:
        bands: list[dict[str, Any]] = []
        for member in sorted(members, key=lambda x: x.band_coord):
            placed = False
            for band in bands:
                if abs(member.band_coord - band["coord"]) <= self.coord_tol:
                    band["members"].append(member)
                    values = [m.band_coord for m in band["members"]]
                    band["coord"] = sum(values) / len(values)
                    placed = True
                    break
            if not placed:
                bands.append({"coord": member.band_coord, "members": [member]})
        return [band["members"] for band in bands]

    def _connected_components(self, members: list[RawMember]) -> list[list[RawMember]]:
        dsu = _DisjointSet(len(members))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if self._interval_gap(members[i].interval, members[j].interval) <= self.coord_tol:
                    dsu.union(i, j)

        groups: dict[int, list[RawMember]] = {}
        for idx, member in enumerate(members):
            root = dsu.find(idx)
            groups.setdefault(root, []).append(member)
        return list(groups.values())

    def _merge_component(
        self,
        members: list[RawMember],
        node_id: int,
        kind: str,
        axis: str,
    ) -> MergedMember:
        band_coord = sum(member.band_coord for member in members) / len(members)
        lo = min(member.interval[0] for member in members)
        hi = max(member.interval[1] for member in members)
        if axis == "horizontal":
            start = (lo, band_coord)
            end = (hi, band_coord)
        else:
            start = (band_coord, lo)
            end = (band_coord, hi)

        return MergedMember(
            node_id=node_id,
            kind=kind,
            axis=axis,
            band_coord=band_coord,
            start=start,
            end=end,
            length=self._distance(start, end),
            raw_member_ids=tuple(sorted(member.raw_id for member in members)),
        )

    def _build_edges(
        self,
        raw_members: list[RawMember],
        raw_to_node: dict[int, int],
    ) -> dict[tuple[int, int], set[tuple[float, float]]]:
        edges: dict[tuple[int, int], set[tuple[float, float]]] = {}

        by_node: dict[int, list[int]] = {}
        for member in raw_members:
            for node in (member.start_node, member.end_node):
                if node is None:
                    continue
                by_node.setdefault(node, []).append(member.raw_id)

        for raw_ids in by_node.values():
            self._connect_raw_ids_by_node(raw_ids, raw_members, raw_to_node, edges)

        # for i in range(len(raw_members)):
        #     for j in range(i + 1, len(raw_members)):
        #         points = self._intersection_points(raw_members[i], raw_members[j])
        #         if points:
        #             self._connect_pair(
        #                 raw_members[i].raw_id, raw_members[j].raw_id, raw_to_node, edges, points
        #             )

        return edges

    def _connect_raw_ids_by_node(
        self,
        raw_ids: list[int],
        raw_members: list[RawMember],
        raw_to_node: dict[int, int],
        edges: dict[tuple[int, int], set[tuple[float, float]]],
    ):
        member_map = {member.raw_id: member for member in raw_members}
        ids = sorted(set(raw_ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = member_map.get(ids[i])
                b = member_map.get(ids[j])
                if a is None or b is None:
                    continue
                points = self._intersection_points(a, b)
                if not points:
                    points = self._shared_endpoints(a, b)
                if points:
                    self._connect_pair(a.raw_id, b.raw_id, raw_to_node, edges, points)

    def _connect_pair(
        self,
        raw_id_a: int,
        raw_id_b: int,
        raw_to_node: dict[int, int],
        edges: dict[tuple[int, int], set[tuple[float, float]]],
        points: list[tuple[float, float]],
    ):
        if raw_id_a not in raw_to_node or raw_id_b not in raw_to_node:
            return
        u = raw_to_node[raw_id_a]
        v = raw_to_node[raw_id_b]
        if u == v:
            return
        key = (u, v) if u < v else (v, u)
        bucket = edges.setdefault(key, set())
        for point in points:
            bucket.add(self._round_point(point))

    def _touches(self, a: RawMember, b: RawMember) -> bool:
        return bool(self._intersection_points(a, b))

    def _intersection_points(self, a: RawMember, b: RawMember) -> list[tuple[float, float]]:
        if a.axis == b.axis:
            return self._collinear_intersection_points(a, b)
        return self._orthogonal_intersection_points(a, b)

    def _collinear_intersection_points(self, a: RawMember, b: RawMember) -> list[tuple[float, float]]:
        if abs(a.band_coord - b.band_coord) > self.coord_tol:
            return []

        overlap_lo = max(a.interval[0], b.interval[0])
        overlap_hi = min(a.interval[1], b.interval[1])
        if overlap_lo <= overlap_hi + self.coord_tol:
            if abs(overlap_hi - overlap_lo) <= self.coord_tol:
                value = (overlap_lo + overlap_hi) / 2.0
                return [self._point_on_axis(a.axis, a.band_coord, value)]
            return [
                self._point_on_axis(a.axis, a.band_coord, overlap_lo),
                self._point_on_axis(a.axis, a.band_coord, overlap_hi),
            ]
        return []

    def _orthogonal_intersection_points(self, a: RawMember, b: RawMember) -> list[tuple[float, float]]:
        h = a if a.axis == "horizontal" else b
        v = b if a.axis == "horizontal" else a
        x = v.band_coord
        y = h.band_coord
        if (
            h.interval[0] - self.coord_tol <= x <= h.interval[1] + self.coord_tol
            and v.interval[0] - self.coord_tol <= y <= v.interval[1] + self.coord_tol
        ):
            return [(x, y)]
        return []

    def _shared_endpoints(self, a: RawMember, b: RawMember) -> list[tuple[float, float]]:
        points = []
        for pa in (a.start, a.end):
            for pb in (b.start, b.end):
                if self._distance(pa, pb) <= self.coord_tol:
                    points.append(((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0))
        return points

    def _member_kind(self, member_type: Any, beam_role: Any) -> str:
        if member_type == "shearwall":
            return "wall"
        if member_type != "beam":
            raise ValueError(f"Unsupported member type: {member_type}")
        if str(beam_role or "primary").lower() == "secondary":
            return "secondary_beam"
        return "primary_beam"

    def _axis_of(self, start: tuple[float, float], end: tuple[float, float]) -> str | None:
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dy <= self.axis_tol and dx > self.axis_tol:
            return "horizontal"
        if dx <= self.axis_tol and dy > self.axis_tol:
            return "vertical"
        return None

    def _band_value(self, start: tuple[float, float], end: tuple[float, float], axis: str) -> float:
        if axis == "horizontal":
            return (start[1] + end[1]) / 2.0
        return (start[0] + end[0]) / 2.0

    def _interval_of(
        self, start: tuple[float, float], end: tuple[float, float], axis: str
    ) -> tuple[float, float]:
        if axis == "horizontal":
            lo, hi = sorted((start[0], end[0]))
        else:
            lo, hi = sorted((start[1], end[1]))
        return lo, hi

    def _interval_gap(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        if a[0] > b[0]:
            a, b = b, a
        return max(0.0, b[0] - a[1])

    def _point(self, value: Any) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError(f"Invalid point: {value}")
        return float(value[0]), float(value[1])

    def _point_on_axis(self, axis: str, band_coord: float, value: float) -> tuple[float, float]:
        if axis == "horizontal":
            return value, band_coord
        return band_coord, value

    def _round_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0], 6), round(point[1], 6)

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


def build_member_graph(
    data: dict[str, Any], coord_tol: float = 0.01, axis_tol: float | None = None
) -> nx.Graph:
    return MemberGraphBuilder(coord_tol=coord_tol, axis_tol=axis_tol).build_graph(data)


def build_member_graph_data(
    data: dict[str, Any],
    coord_tol: float = 0.01,
    axis_tol: float | None = None,
) -> dict[str, Any]:
    return MemberGraphBuilder(coord_tol=coord_tol, axis_tol=axis_tol).build_graph_data(data)
