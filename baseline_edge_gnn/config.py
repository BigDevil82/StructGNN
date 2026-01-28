"""
Configuration for Edge-based GNN Baseline
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EdgeDataConfig:
    """Data configuration"""

    # Data paths
    JSON_PATH: str = (
        r"E:\Common\Desktop\Research\deepLearning\codes\StructGAN_v1\processed\shear_wall_dataset.json"  # Path to pre-processed JSON
    )
    TRAIN_DIR: str = (
        r"dxf\shearwall_split_8_2\train"  # Directory containing train DXF files (use filenames as keys)
    )
    TEST_DIR: str = (
        r"dxf\shearwall_split_8_2\test"  # Directory containing test DXF files (use filenames as keys)
    )
    CACHE_DIR: str = "data_cache/baseline_GNN-EP4-CV"

    # Normalization scale (in mm)
    NORM_SCALE: float = 2088.0

    # Minimum edge length to consider (mm)
    MIN_EDGE_LENGTH: float = 10.0

    # Tolerance for point merging (mm)
    POINT_MERGE_TOL: float = 3.0

    # Data augmentation modes (same as your method for fair comparison)
    # Added translate_1~4 for translation augmentation (position invariance)
    AUGMENTATIONS: List[str] = field(
        default_factory=lambda: [
            "none",
            "flip_x",
            "flip_y",
            "rot_90",
            "rot_180",
            "rot_270",
            "translate_1",
            "translate_2",
            "translate_3",
            "translate_4",
        ]
    )
    # ["none", "flip_x", "flip_y", "rot_90", "rot_180", "rot_270"]

    # Edge types
    EDGE_TYPE_PSW: int = 0  # Potential Shear Wall
    EDGE_TYPE_DOOR: int = 1
    EDGE_TYPE_WINDOW: int = 2


@dataclass
class EdgeModelConfig:
    """Model configuration for GNN-EP-4"""

    # Node feature dimension (normalized x, y) - REMOVED category to avoid bias
    NODE_IN_DIM: int = 2

    # Edge feature dimension: 3 (one-hot type) + 4 (coordinates) + 1 (length) = 8
    # REMOVED category to avoid category bias
    EDGE_IN_DIM: int = 8

    # Hidden dimensions for SFLayers
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [16, 16, 32, 32, 32, 32])

    # Output dimension of GNN backbone
    GNN_OUT_DIM: int = 32

    # MLP hidden dimensions
    MLP_DIMS: List[int] = field(default_factory=lambda: [32, 32, 16, 16, 8, 2])

    # Dropout rate
    DROPOUT: float = 0.25

    # Output dimension (left ratio, right ratio)
    OUT_DIM: int = 2


@dataclass
class EdgeTrainConfig:
    """Training configuration"""

    # Training parameters
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    EPOCHS: int = 200

    # Early stopping
    PATIENCE: int = 30

    # Device
    DEVICE: str = "cuda"

    # Random seed
    SEED: int = 42

    # Model save path
    MODEL_SAVE_PATH: str = "result/shearwall_pred/baseline-GNN-EP4-CV/baseline_edge_gnn.pt"


# Global config instances
data_config = EdgeDataConfig()
model_config = EdgeModelConfig()
train_config = EdgeTrainConfig()
