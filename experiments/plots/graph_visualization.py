"""
Graph visualization utilities for room-based vs edge-based representations.

Saves two visualizations for a specified DXF:
1) Room-based graph (rooms as nodes)
2) Edge-based graph (component intersections as nodes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from experiments.research.baseline_edge_gnn.config import data_config as edge_data_config
from experiments.research.baseline_edge_gnn.graph_builder import build_graph_from_json, load_json_data
from experiments.research.baseline_edge_gnn.utils import visualize_edge_graph
from src.data_engine.preprocess.layout_graph import convert_to_graph


def _layout_style() -> dict:
    """Style settings aligned to preprocess.layout_graph.LayoutGraphBuilder.visualize."""
    return {
        "edge_colors": {
            edge_data_config.EDGE_TYPE_PSW: "gray",
            edge_data_config.EDGE_TYPE_DOOR: "blue",
            edge_data_config.EDGE_TYPE_WINDOW: "green",
        },
        "edge_labels": {
            edge_data_config.EDGE_TYPE_PSW: "PSW",
            edge_data_config.EDGE_TYPE_DOOR: "Door",
            edge_data_config.EDGE_TYPE_WINDOW: "Window",
        },
        "edge_width": 3.5,
        "edge_alpha": 0.6,
        "draw_sw": True,
        "sw_color": "red",
        "sw_width": 4,
        "sw_alpha": 0.7,
        "show_legend": False,
        "show_grid": False,
        "axis_off": True,
        "node_color": "green",
        "node_size": 100,
        "node_edge_color": "black",
        "node_edge_width": 1.0,
        "node_labels": False,
        "node_label_color": "black",
        "node_label_size": 12,
        "node_label_weight": "bold",
    }


def visualize_edge_graph_from_json(
    json_path: str,
    file_key: str,
    save_path: str,
    title: str = "Edge Graph Structure",
) -> None:
    """
    Visualize edge-based graph from JSON data for a given key.

    Args:
        json_path: JSON file containing edge segments.
        file_key: Key in JSON (usually DXF filename without suffix).
        save_path: Output image path.
        title: Figure title.
    """
    json_data = load_json_data(json_path)
    builder = build_graph_from_json(json_data, file_key, mode="none")
    if builder is None:
        raise ValueError(f"No edge graph data for key: {file_key}")

    data = builder.to_pyg_data()
    if data is None:
        raise ValueError(f"Failed to build edge graph data for key: {file_key}")

    print("Node and Edge counts:", len(builder.nodes), len(builder.edges))
    visualize_edge_graph(
        data=data,
        nodes=builder.nodes,
        edges=builder.edges,
        save_path=save_path,
        title=title,
        style=_layout_style(),
    )


def visualize_room_graph(
    dxf_path: str,
    save_path: str,
) -> None:
    """Visualize room-based graph (rooms as nodes)."""
    convert_to_graph(dxf_path, save_path=save_path)


def visualize_both(
    dxf_path: str,
    json_path: Optional[str] = None,
    output_dir: str = "experiments/plots",
) -> tuple[str, str]:
    """
    Save both room-based and edge-based graph visualizations for a DXF.

    The edge-based key is derived from the DXF filename (without suffix).

    Returns:
        (room_graph_path, edge_graph_path)
    """
    dxf_path_obj = Path(dxf_path)
    file_key = dxf_path_obj.stem

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    room_graph_path = output_dir_path / f"{file_key}_room_graph.png"
    edge_graph_path = output_dir_path / f"{file_key}_edge_graph.png"

    visualize_room_graph(str(dxf_path_obj), str(room_graph_path))

    json_path = json_path or edge_data_config.JSON_PATH
    visualize_edge_graph_from_json(
        json_path=json_path,
        file_key=file_key,
        save_path=str(edge_graph_path),
        title="Edge Graph Structure",
    )

    return str(room_graph_path), str(edge_graph_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize room vs edge graph for a DXF")
    parser.add_argument("--dxf", required=True, help="DXF file path")
    parser.add_argument(
        "--json",
        default=None,
        help="JSON file path (defaults to baseline_edge_gnn.config.data_config.JSON_PATH)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/result/paper_plots", help="Output directory for images"
    )

    args = parser.parse_args()

    room_path, edge_path = visualize_both(args.dxf, json_path=args.json, output_dir=args.output_dir)
    print(f"Saved room graph: {room_path}")
    print(f"Saved edge graph: {edge_path}")
