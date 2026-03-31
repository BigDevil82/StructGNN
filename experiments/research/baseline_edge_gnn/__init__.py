"""
Baseline: Edge-based GNN for Shear Wall Prediction

Implementation based on:
"Intelligent design of shear wall layout based on graph neural networks"

This approach uses graph edge representation where:
- Nodes: Intersection points of walls, doors, windows
- Edges: Building components (Potential Shear Walls, Doors, Windows)
- Prediction: Shear wall length ratios at both ends of each PSW edge
"""

from .graph_builder import EdgeGraphBuilder
from .dataset import EdgeShearWallDataset
from .model import GNNEP4

__all__ = ["EdgeGraphBuilder", "EdgeShearWallDataset", "GNNEP4"]
