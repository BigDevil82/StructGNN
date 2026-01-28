"""
Edge-based Graph Builder (JSON Format)

Builds graph structure from pre-processed JSON data where:
- Nodes: Intersection points of building components (merged by tolerance)
- Edges: Wall segments, doors, windows with shear wall ratios

JSON format:
{
    "file_name": {
        "walls": [{"start": [x, y], "end": [x, y], "left_shear_ratio": 0.5, "right_shear_ratio": 0.5}, ...],
        "windows": [...],
        "doors": [...]
    }
}
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from shapely.geometry import LineString
from torch_geometric.data import Data

from baseline_edge_gnn.config import data_config, model_config


class EdgeGraphBuilder:
    """
    Builds edge-based graph from JSON segment data.

    Handles coordinate precision errors by merging nearby points.
    """

    def __init__(
        self,
        walls: List[Dict],
        windows: List[Dict],
        doors: List[Dict],
        norm_scale: float = None,
        point_merge_tol: float = None,
        category: int = -1,
    ):
        """
        Args:
            walls: List of wall segments with shear ratios
            windows: List of window segments
            doors: List of door segments
            norm_scale: Normalization scale (default from config)
            point_merge_tol: Tolerance for merging nearby points (default from config)
            category: Design category (0: L17, 1: L27, 2: L1L28, -1: unknown)
        """
        self.walls = walls
        self.windows = windows
        self.doors = doors
        self.category = category

        self.norm_scale = norm_scale or data_config.NORM_SCALE
        self.point_merge_tol = point_merge_tol or data_config.POINT_MERGE_TOL

        # Graph components
        self.nodes: List[Tuple[float, float]] = []
        self.node_to_idx: Dict[int, int] = {}  # hash -> index
        self.edges: List[Dict] = []

        # Centroid for centering (computed after building graph)
        self.centroid_x: float = 0.0
        self.centroid_y: float = 0.0

        # Build the graph
        self._build_graph()

        # Compute centroid for centering normalization
        self._compute_centroid()

    def _point_hash(self, x: float, y: float) -> int:
        """Create a hash for a point based on grid cell."""
        # Use grid-based hashing for fast lookup
        grid_x = int(round(x / self.point_merge_tol))
        grid_y = int(round(y / self.point_merge_tol))
        return hash((grid_x, grid_y))

    def _find_or_add_node(self, x: float, y: float) -> int:
        """
        Find existing node within tolerance or add new node.

        Uses grid-based hashing for efficiency, then checks nearby cells.
        """
        # Check nearby grid cells
        grid_x = int(round(x / self.point_merge_tol))
        grid_y = int(round(y / self.point_merge_tol))

        # Check 3x3 neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_hash = hash((grid_x + dx, grid_y + dy))
                if cell_hash in self.node_to_idx:
                    idx = self.node_to_idx[cell_hash]
                    nx, ny = self.nodes[idx]
                    dist = np.sqrt((x - nx) ** 2 + (y - ny) ** 2)
                    if dist < self.point_merge_tol:
                        return idx

        # Add new node
        idx = len(self.nodes)
        self.nodes.append((x, y))
        self.node_to_idx[hash((grid_x, grid_y))] = idx
        return idx

    def _add_edge(
        self,
        start: List[float],
        end: List[float],
        edge_type: int,
        left_ratio: float = 0.0,
        right_ratio: float = 0.0,
    ):
        """Add an edge with its features."""
        start_idx = self._find_or_add_node(start[0], start[1])
        end_idx = self._find_or_add_node(end[0], end[1])

        if start_idx == end_idx:
            return

        # Compute edge length
        length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        if length < data_config.MIN_EDGE_LENGTH:
            return

        line = LineString([start, end])

        self.edges.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "edge_type": edge_type,
                "line": line,
                "length": length,
                "sw_ratio_start": left_ratio,
                "sw_ratio_end": right_ratio,
            }
        )

    def _build_graph(self):
        """Build graph from all segments."""
        # Process walls (PSW - Potential Shear Walls)
        for wall in self.walls:
            self._add_edge(
                wall["start"],
                wall["end"],
                data_config.EDGE_TYPE_PSW,
                wall.get("left_shear_ratio", 0.0),
                wall.get("right_shear_ratio", 0.0),
            )

        # Process windows
        for window in self.windows:
            self._add_edge(
                window["start"],
                window["end"],
                data_config.EDGE_TYPE_WINDOW,
                0.0,
                0.0,
            )

        # Process doors
        for door in self.doors:
            self._add_edge(
                door["start"],
                door["end"],
                data_config.EDGE_TYPE_DOOR,
                0.0,
                0.0,
            )

    def _compute_centroid(self):
        """Compute centroid of all nodes for centering."""
        if len(self.nodes) == 0:
            return
        xs = [n[0] for n in self.nodes]
        ys = [n[1] for n in self.nodes]
        self.centroid_x = sum(xs) / len(xs)
        self.centroid_y = sum(ys) / len(ys)

    def _normalize_coord(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize coordinates: center at origin, then scale."""
        # Center the coordinates
        centered_x = x - self.centroid_x
        centered_y = y - self.centroid_y
        # Scale to normalized range
        return (centered_x / self.norm_scale, centered_y / self.norm_scale)

    def to_pyg_data(self) -> Optional[Data]:
        """
        Convert graph to PyTorch Geometric Data object.

        Node features: [norm_x, norm_y] (2 dim) - REMOVED category to avoid category bias
        Edge features: [type_onehot(3), start_x, start_y, end_x, end_y, length] (8 dim) - REMOVED category
        Edge labels: [ratio_start, ratio_end] (2 dim), only for PSW edges
        """
        if len(self.nodes) == 0 or len(self.edges) == 0:
            return None

        # Node features: [norm_x, norm_y] - REMOVED category one-hot
        node_features = []
        for x, y in self.nodes:
            norm_x, norm_y = self._normalize_coord(x, y)
            node_features.append([norm_x, norm_y])

        # Edge index and features (bidirectional)
        edge_index = []
        edge_attr = []
        edge_labels = []
        edge_mask = []  # 1 for PSW edges, 0 for others

        for edge in self.edges:
            start_idx = edge["start_idx"]
            end_idx = edge["end_idx"]
            edge_type = edge["edge_type"]

            # Add bidirectional edges
            for src, dst, ratio_s, ratio_e in [
                (start_idx, end_idx, edge["sw_ratio_start"], edge["sw_ratio_end"]),
                (end_idx, start_idx, edge["sw_ratio_end"], edge["sw_ratio_start"]),
            ]:
                edge_index.append([src, dst])

                # One-hot edge type (4 classes)
                type_onehot = [0.0] * 3
                type_onehot[edge_type] = 1.0

                # Normalized coordinates
                src_x, src_y = self._normalize_coord(*self.nodes[src])
                dst_x, dst_y = self._normalize_coord(*self.nodes[dst])

                # Normalized length
                norm_length = edge["length"] / self.norm_scale

                # Edge feature vector: type_onehot(3) + coords(4) + length(1) = 8
                # REMOVED category one-hot to avoid category bias
                feat = type_onehot + [src_x, src_y, dst_x, dst_y, norm_length]
                edge_attr.append(feat)

                # Labels
                edge_labels.append([ratio_s, ratio_e])

                # Mask (only compute loss for PSW edges)
                edge_mask.append(1.0 if edge_type == data_config.EDGE_TYPE_PSW else 0.0)

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        edge_labels_tensor = torch.tensor(edge_labels, dtype=torch.float)
        edge_mask_tensor = torch.tensor(edge_mask, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=edge_labels_tensor,
            edge_mask=edge_mask_tensor,
        )

        return data

    @staticmethod
    def apply_augmentation(
        walls: List[Dict],
        windows: List[Dict],
        doors: List[Dict],
        mode: str,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Apply geometric augmentation to all segments.

        Args:
            mode: 'none', 'flip_x', 'flip_y', 'rot_90', 'rot_180', 'rot_270',
                  'translate_1', 'translate_2', ... (random translations)

        Returns:
            Transformed walls, windows, doors
        """
        if mode == "none":
            return walls, windows, doors

        # Handle translation augmentation
        if mode.startswith("translate"):
            import random

            # Random translation offset (within a reasonable range)
            tx = random.uniform(-500, 500)
            ty = random.uniform(-500, 500)

            def translate_point(x: float, y: float) -> Tuple[float, float]:
                return (x + tx, y + ty)

            def translate_segment(seg: Dict) -> Dict:
                new_start = translate_point(seg["start"][0], seg["start"][1])
                new_end = translate_point(seg["end"][0], seg["end"][1])
                result = {
                    "start": list(new_start),
                    "end": list(new_end),
                }
                if "left_shear_ratio" in seg:
                    result["left_shear_ratio"] = seg["left_shear_ratio"]
                    result["right_shear_ratio"] = seg["right_shear_ratio"]
                return result

            return (
                [translate_segment(w) for w in walls],
                [translate_segment(w) for w in windows],
                [translate_segment(d) for d in doors],
            )

        def transform_point(x: float, y: float) -> Tuple[float, float]:
            if mode == "flip_x":
                return (-x, y)
            elif mode == "flip_y":
                return (x, -y)
            elif mode == "rot_90":
                return (-y, x)
            elif mode == "rot_180":
                return (-x, -y)
            elif mode == "rot_270":
                return (y, -x)
            return (x, y)

        def transform_segment(seg: Dict) -> Dict:
            new_start = transform_point(seg["start"][0], seg["start"][1])
            new_end = transform_point(seg["end"][0], seg["end"][1])

            result = {
                "start": list(new_start),
                "end": list(new_end),
            }

            # Preserve shear ratios
            if "left_shear_ratio" in seg:
                result["left_shear_ratio"] = seg["left_shear_ratio"]
                result["right_shear_ratio"] = seg["right_shear_ratio"]

            return result

        return (
            [transform_segment(w) for w in walls],
            [transform_segment(w) for w in windows],
            [transform_segment(d) for d in doors],
        )


def load_json_data(json_path: str) -> Dict:
    """Load JSON data from file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_category(file_key: str) -> int:
    """
    Get category index from file key.

    Categories:
    - 0: L17 (low shear wall ratio)
    - 1: L27 (medium shear wall ratio)
    - 2: L1L28 (high shear wall ratio)
    """
    if file_key.startswith("L17"):
        return 0
    elif file_key.startswith("L27"):
        return 1
    elif file_key.startswith("L1L28"):
        return 2
    return -1


def build_graph_from_json(
    json_data: Dict,
    file_key: str,
    mode: str = "none",
) -> Optional[EdgeGraphBuilder]:
    """
    Build edge graph from JSON data for a specific file.

    Args:
        json_data: Full JSON dict
        file_key: Key for the specific file (e.g., "L17_130")
        mode: Augmentation mode

    Returns:
        EdgeGraphBuilder instance or None
    """
    if file_key not in json_data:
        return None

    file_data = json_data[file_key]

    walls = file_data.get("walls", [])
    windows = file_data.get("windows", [])
    doors = file_data.get("doors", [])

    if not walls and not windows and not doors:
        return None

    # Get category from file key
    category = get_file_category(file_key)

    # Apply augmentation
    walls, windows, doors = EdgeGraphBuilder.apply_augmentation(walls, windows, doors, mode)

    # Build graph with category
    builder = EdgeGraphBuilder(walls, windows, doors, category=category)

    return builder
