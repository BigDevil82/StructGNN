from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data_engine.postprocess.member_graph import MemberGraphBuilder

KIND_TO_ID = {
    "wall": 0,
    "primary_beam": 1,
    "secondary_beam": 2,
}
AXIS_TO_ID = {
    "horizontal": 0,
    "vertical": 1,
}


@dataclass(frozen=True)
class LayoutGraphCacheConfig:
    layout_json_dir: str = r"data\dxf\cad_json_data\fem_raw"
    output_dir: str = r"data\parametric\surrogate_dataset\gnn_graph_cache"
    coord_tol: float = 0.01


def build_layout_graph_cache(cfg: LayoutGraphCacheConfig) -> dict[str, int]:
    src_dir = Path(cfg.layout_json_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(src_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No layout json found in {src_dir}")

    builder = MemberGraphBuilder(coord_tol=cfg.coord_tol)
    num_ok = 0
    num_skip = 0

    for path in tqdm(paths, desc="[surrogate][gnn] build graph cache"):
        layout_id = path.stem
        out_path = out_dir / f"{layout_id}.pt"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            graph_data = _build_layout_graph_data(raw, builder)
            torch.save(graph_data, out_path)
            num_ok += 1
        except Exception:
            num_skip += 1

    summary = {
        "total_layouts": len(paths),
        "cached_layouts": num_ok,
        "skipped_layouts": num_skip,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _build_layout_graph_data(raw: dict, builder: MemberGraphBuilder) -> dict[str, torch.Tensor]:
    graph = builder.build_graph(raw)
    if graph.number_of_nodes() == 0:
        raise ValueError("empty graph")

    node_ids = sorted(graph.nodes())
    id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

    xs = []
    centers: list[tuple[float, float]] = []
    lengths = []

    all_points = []
    for _, attrs in graph.nodes(data=True):
        sx, sy = attrs["start"]
        ex, ey = attrs["end"]
        all_points.extend([(sx, sy), (ex, ey)])

    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    diag = float(np.sqrt(dx * dx + dy * dy))

    for node_id in node_ids:
        attrs = graph.nodes[node_id]
        sx, sy = attrs["start"]
        ex, ey = attrs["end"]
        cx = 0.5 * (sx + ex)
        cy = 0.5 * (sy + ey)
        length = float(attrs.get("length", np.hypot(ex - sx, ey - sy)))

        centers.append((cx, cy))
        lengths.append(length)

        kind_onehot = _onehot(KIND_TO_ID.get(attrs.get("kind", "wall"), 0), 3)
        axis_onehot = _onehot(AXIS_TO_ID.get(attrs.get("axis", "horizontal"), 0), 2)
        cx_norm = (cx - min_x) / dx
        cy_norm = (cy - min_y) / dy
        len_norm = length / max(diag, 1e-6)

        boundary = float((cx_norm < 0.05) or (cx_norm > 0.95) or (cy_norm < 0.05) or (cy_norm > 0.95))
        dist_center = float(np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2))

        xs.append(kind_onehot + axis_onehot + [len_norm, cx_norm, cy_norm, boundary, dist_center])

    edge_index = []
    edge_attr = []
    for u, v in graph.edges():
        ui = id_map[u]
        vi = id_map[v]
        et = _edge_type(graph.nodes[u].get("kind", "wall"), graph.nodes[v].get("kind", "wall"))
        ux, uy = centers[ui]
        vx, vy = centers[vi]
        dist = float(np.hypot(ux - vx, uy - vy) / max(diag, 1e-6))
        axis_u = graph.nodes[u].get("axis", "horizontal")
        axis_v = graph.nodes[v].get("axis", "horizontal")
        orth = float(axis_u != axis_v)
        feat = _onehot(et, 4) + [dist, orth]

        edge_index.append([ui, vi])
        edge_index.append([vi, ui])
        edge_attr.append(feat)
        edge_attr.append(feat)

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    wall_count = sum(1 for _, a in graph.nodes(data=True) if a.get("kind") == "wall")
    pbeam_count = sum(1 for _, a in graph.nodes(data=True) if a.get("kind") == "primary_beam")
    sbeam_count = sum(1 for _, a in graph.nodes(data=True) if a.get("kind") == "secondary_beam")

    graph_feat = np.array(
        [
            dx / max(dy, 1e-6),
            float(graph.number_of_nodes()),
            float(graph.number_of_edges()),
            float(wall_count),
            float(pbeam_count),
            float(sbeam_count),
            float(np.sum(lengths)),
        ],
        dtype=np.float32,
    )

    return {
        "x": torch.tensor(np.asarray(xs, dtype=np.float32)),
        "edge_index": torch.tensor(np.asarray(edge_index, dtype=np.int64)).t().contiguous(),
        "edge_attr": torch.tensor(np.asarray(edge_attr, dtype=np.float32)),
        "graph_feat": torch.tensor(graph_feat[None, :]),
    }


def _onehot(idx: int, size: int) -> list[float]:
    out = [0.0] * size
    out[int(max(0, min(size - 1, idx)))] = 1.0
    return out


def _edge_type(kind_u: str, kind_v: str) -> int:
    kinds = {kind_u, kind_v}
    if kinds == {"wall"}:
        return 0
    if "wall" in kinds and "primary_beam" in kinds:
        return 1
    if "wall" in kinds and "secondary_beam" in kinds:
        return 2
    return 3
