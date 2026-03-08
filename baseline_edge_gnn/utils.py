"""
Utility functions for Edge-based GNN Baseline
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import LineString, Polygon
from torch_geometric.data import Data

from baseline_edge_gnn.config import data_config


def visualize_edge_graph(
    data: Data,
    nodes: List[Tuple[float, float]],
    edges: List[dict],
    save_path: str = None,
    title: str = "Edge-based Graph",
    style: Optional[dict] = None,
):
    """
    Visualize the edge-based graph structure.

    Args:
        data: PyG Data object
        nodes: List of (x, y) node coordinates
        edges: List of edge dictionaries
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    if style is None:
        style = {}

    # Color mapping for edge types
    edge_colors = style.get(
        "edge_colors",
        {
            data_config.EDGE_TYPE_PSW: "gray",
            data_config.EDGE_TYPE_DOOR: "blue",
            data_config.EDGE_TYPE_WINDOW: "green",
        },
    )

    edge_labels = style.get(
        "edge_labels",
        {
            data_config.EDGE_TYPE_PSW: "PSW",
            data_config.EDGE_TYPE_DOOR: "Door",
            data_config.EDGE_TYPE_WINDOW: "Window",
        },
    )

    edge_width = style.get("edge_width", 2)
    edge_alpha = style.get("edge_alpha", 0.8)
    draw_sw = style.get("draw_sw", True)
    sw_color = style.get("sw_color", "red")
    sw_width = style.get("sw_width", 4)
    sw_alpha = style.get("sw_alpha", 0.7)
    show_legend = style.get("show_legend", True)
    show_grid = style.get("show_grid", True)
    axis_off = style.get("axis_off", False)
    node_color = style.get("node_color", "black")
    node_size = style.get("node_size", 50)
    node_edge_color = style.get("node_edge_color", None)
    node_edge_width = style.get("node_edge_width", 1)
    node_labels = style.get("node_labels", True)
    node_label_color = style.get("node_label_color", "black")
    node_label_size = style.get("node_label_size", 8)
    node_label_weight = style.get("node_label_weight", "normal")

    # Plot edges
    plotted_types = set()
    for edge in edges:
        line = edge["line"]
        edge_type = edge["edge_type"]
        color = edge_colors.get(edge_type, "black")
        label = edge_labels.get(edge_type, "Unknown") if edge_type not in plotted_types else None

        coords = list(line.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        ax.plot(xs, ys, color=color, linewidth=edge_width, label=label, alpha=edge_alpha)

        # Mark shear wall ratios for PSW edges
        if draw_sw and edge_type == data_config.EDGE_TYPE_PSW:
            ratio_start = edge["sw_ratio_start"]
            ratio_end = edge["sw_ratio_end"]

            if ratio_start > 0:
                # Draw shear wall segment at start
                total_len = line.length
                sw_len = total_len * ratio_start
                sw_point = line.interpolate(sw_len)
                ax.plot(
                    [coords[0][0], sw_point.x],
                    [coords[0][1], sw_point.y],
                    color=sw_color,
                    linewidth=sw_width,
                    alpha=sw_alpha,
                )

            if ratio_end > 0:
                # Draw shear wall segment at end
                total_len = line.length
                sw_len = total_len * ratio_end
                sw_point = line.interpolate(total_len - sw_len)
                ax.plot(
                    [coords[-1][0], sw_point.x],
                    [coords[-1][1], sw_point.y],
                    color=sw_color,
                    linewidth=sw_width,
                    alpha=sw_alpha,
                )

        plotted_types.add(edge_type)

    # Plot nodes
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        c=node_color,
        s=node_size,
        zorder=5,
        label="Nodes" if show_legend else None,
        edgecolors=node_edge_color,
        linewidths=node_edge_width,
    )

    # Add node indices
    if node_labels:
        for i, (x, y) in enumerate(nodes):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=node_label_size,
                ha="center",
                va="center",
                color=node_label_color,
                fontweight=node_label_weight,
            )

    ax.set_aspect("equal")
    if show_legend:
        ax.legend(loc="upper right")
    ax.set_title(title)
    if show_grid:
        ax.grid(True, alpha=0.3)
    if axis_off:
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def visualize_predictions(
    data: Data,
    predictions: torch.Tensor,
    nodes: List[Tuple[float, float]],
    edges: List[dict],
    save_path: str = None,
    title: str = "Prediction vs Ground Truth",
):
    """
    Visualize predicted vs ground truth shear wall ratios.

    Args:
        data: PyG Data object
        predictions: Model predictions [num_edges, 2]
        nodes: List of (x, y) node coordinates
        edges: List of edge dictionaries
        save_path: Path to save figure
        title: Figure title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax_idx, (ax, values, subtitle) in enumerate(
        [
            (axes[0], data.y.cpu().numpy(), "Ground Truth"),
            (
                axes[1],
                predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions,
                "Prediction",
            ),
        ]
    ):
        # Plot edges
        edge_idx = 0
        for edge in edges:
            if edge["edge_type"] != data_config.EDGE_TYPE_PSW:
                edge_idx += 2  # Skip both directions
                continue

            line = edge["line"]
            coords = list(line.coords)

            # Draw base line
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color="lightblue", linewidth=2, alpha=0.5)

            # Get ratios for this edge (forward direction)
            if edge_idx < len(values):
                ratio_start, ratio_end = values[edge_idx]

                if ratio_start > 0.05:
                    total_len = line.length
                    sw_len = total_len * ratio_start
                    sw_point = line.interpolate(sw_len)
                    ax.plot(
                        [coords[0][0], sw_point.x],
                        [coords[0][1], sw_point.y],
                        color="red",
                        linewidth=4,
                        alpha=0.8,
                    )

                if ratio_end > 0.05:
                    total_len = line.length
                    sw_len = total_len * ratio_end
                    sw_point = line.interpolate(total_len - sw_len)
                    ax.plot(
                        [coords[-1][0], sw_point.x],
                        [coords[-1][1], sw_point.y],
                        color="red",
                        linewidth=4,
                        alpha=0.8,
                    )

            edge_idx += 2  # Account for both directions

        # Plot nodes
        node_xs = [n[0] for n in nodes]
        node_ys = [n[1] for n in nodes]
        ax.scatter(node_xs, node_ys, c="black", s=30, zorder=5)

        ax.set_aspect("equal")
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_methods(
    room_based_results: dict,
    edge_based_results: dict,
    save_path: str = None,
):
    """
    Compare results between room-based and edge-based methods.

    Args:
        room_based_results: Results from your method
        edge_based_results: Results from baseline method
        save_path: Path to save comparison figure
    """
    metrics = ["MAE", "Accuracy", "Precision", "Recall"]

    room_values = [
        room_based_results.get("mae", 0),
        room_based_results.get("accuracy", 0),
        room_based_results.get("precision", 0),
        room_based_results.get("recall", 0),
    ]

    edge_values = [
        edge_based_results.get("mae", 0),
        edge_based_results.get("accuracy", 0),
        edge_based_results.get("precision", 0),
        edge_based_results.get("recall", 0),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, room_values, width, label="Room-based (Ours)", color="steelblue")
    bars2 = ax.bar(x + width / 2, edge_values, width, label="Edge-based (Baseline)", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Method Comparison: Room-based vs Edge-based GNN")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def get_file_category(filename: str) -> int:
    """
    Get category index from filename.

    Categories:
    - 0: L17 (low shear wall ratio)
    - 1: L1L28 (medium shear wall ratio)
    - 2: L29L56 (high shear wall ratio)

    Args:
        filename: DXF filename

    Returns:
        Category index (0, 1, 2) or -1 if unknown
    """
    if filename.startswith("L17"):
        return 0
    elif filename.startswith("L1L28"):
        return 1
    elif filename.startswith("L29L56"):
        return 2
    return -1
