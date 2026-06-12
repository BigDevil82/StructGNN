"""
Visualization for Edge-based GNN Baseline (JSON Format)

Visualizes predictions on test samples:
1. Shows edge-based graph structure with walls, doors, windows
2. Compares predicted vs ground truth shear wall ratios
3. Saves results for each test sample
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docs.paper.experiments.research.baseline_edge_gnn.config import data_config, model_config, train_config
from docs.paper.experiments.research.baseline_edge_gnn.ensemble_model import EnsembleGNNEP4
from docs.paper.experiments.research.baseline_edge_gnn.graph_builder import (
    EdgeGraphBuilder,
    build_graph_from_json,
    load_json_data,
)
from docs.paper.experiments.research.baseline_edge_gnn.model import GNNEP4, compute_metrics


def load_model(model_path: str, device: str = "cuda") -> GNNEP4:
    """Load trained model from checkpoint."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = GNNEP4(
        node_in_dim=model_config.NODE_IN_DIM,
        edge_in_dim=model_config.EDGE_IN_DIM,
        gnn_out_dim=model_config.GNN_OUT_DIM,
        dropout=model_config.DROPOUT,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model


def load_ensemble_model(cv_dir: str, n_folds: int = None, device: str = "cuda") -> EnsembleGNNEP4:
    """Load ensemble model from cross-validation directory."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    device_obj = torch.device(device)

    ensemble_model = EnsembleGNNEP4.from_cv_dir(
        cv_dir=cv_dir,
        node_in_dim=model_config.NODE_IN_DIM,
        edge_in_dim=model_config.EDGE_IN_DIM,
        gnn_out_dim=model_config.GNN_OUT_DIM,
        dropout=model_config.DROPOUT,
        n_folds=n_folds,
        device=device_obj,
    )

    return ensemble_model


def predict_single_sample(
    json_data: Dict,
    file_key: str,
    model: GNNEP4,
    device: str = "cuda",
) -> Tuple[Optional[EdgeGraphBuilder], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run prediction on a single sample.

    Returns:
        builder: EdgeGraphBuilder with graph structure
        predictions: Predicted ratios [num_edges * 2, 2] (bidirectional)
        ground_truth: Ground truth ratios [num_edges * 2, 2]
    """
    builder = build_graph_from_json(json_data, file_key, mode="none")
    if builder is None:
        return None, None, None

    data = builder.to_pyg_data()
    if data is None:
        return None, None, None

    batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        pred = model(batch)

    predictions = pred.cpu().numpy()
    # 将小于0.1的预测值设为0，避免显示过短的剪力墙
    predictions[predictions < 0.1] = 0.0
    ground_truth = data.y.cpu().numpy()

    return builder, predictions, ground_truth


def visualize_prediction(
    builder: EdgeGraphBuilder,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    file_key: str,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Visualize prediction results for a single file.

    Shows:
    - Left: Ground truth shear wall distribution
    - Right: Predicted shear wall distribution
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    for ax, values, title in [
        (axes[0], ground_truth, "Ground Truth"),
        (axes[1], predictions, "Prediction"),
    ]:
        edge_idx = 0

        for edge in builder.edges:
            line = edge["line"]
            edge_type = edge["edge_type"]
            coords = list(line.coords)

            # Only process PSW edges (skip doors and windows)
            if edge_type == data_config.EDGE_TYPE_PSW and edge_idx < len(values):
                ratio_start, ratio_end = values[edge_idx]

                # Calculate non-shear wall segments (填充墙)
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]

                # Draw full edge as fill wall first (灰色填充墙，与shearwall_pred保持一致)
                ax.plot(xs, ys, color="#BBBBBB", linewidth=8, zorder=3)

                # Draw shear wall overlay on top (红色剪力墙)
                # Shear wall at start (left)
                color = "green" if title == "Ground Truth" else "red"
                if ratio_start > 0.02:
                    total_len = line.length
                    sw_len = total_len * ratio_start
                    sw_point = line.interpolate(sw_len)
                    ax.plot(
                        [coords[0][0], sw_point.x],
                        [coords[0][1], sw_point.y],
                        color=color,
                        linewidth=4,
                        zorder=4,
                    )

                # Shear wall at end (right)
                if ratio_end > 0.02:
                    total_len = line.length
                    sw_len = total_len * ratio_end
                    sw_point = line.interpolate(total_len - sw_len)
                    ax.plot(
                        [coords[-1][0], sw_point.x],
                        [coords[-1][1], sw_point.y],
                        color=color,
                        linewidth=4,
                        zorder=4,
                    )

            edge_idx += 2  # Account for bidirectional edges

        ax.set_aspect("equal")
        # ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    # Calculate metrics for PSW edges
    mask = torch.tensor(
        [1.0 if e["edge_type"] == data_config.EDGE_TYPE_PSW else 0.0 for e in builder.edges for _ in range(2)]
    )
    pred_tensor = torch.tensor(predictions)
    gt_tensor = torch.tensor(ground_truth)
    metrics = compute_metrics(pred_tensor, gt_tensor, mask)

    plt.suptitle(
        f"{file_key} MAE: {metrics['mae']:.4f} | Accuracy: {metrics['accuracy']:.4f}",
        fontsize=16,
    )
    plt.tight_layout()

    # Apply horizontal flip to both axes
    for ax in axes:
        ax.invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return metrics


def run_test_visualization(
    model_path: str = None,
    cv_dir: str = None,
    n_folds: int = None,
    json_path: str = None,
    output_dir: str = None,
    file_keys: List[str] = None,
    max_samples: int = None,
    device: str = "cuda",
):
    """
    Run visualization on test samples.

    Args:
        model_path: Path to trained model (for single model)
        cv_dir: Directory containing fold models (for ensemble)
        n_folds: Number of folds for ensemble
        json_path: Path to JSON data file
        output_dir: Directory to save visualizations
        file_keys: Specific file keys to visualize (None for all)
        max_samples: Maximum samples to visualize
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model (ensemble or single)
    if cv_dir is not None:
        print(f"\nLoading ensemble model from: {cv_dir}")
        model = load_ensemble_model(cv_dir, n_folds, device)
        model_type = "Ensemble"
    elif model_path is not None:
        print(f"\nLoading single model from: {model_path}")
        model = load_model(model_path, device)
        model_type = "Single"
    else:
        raise ValueError("Either model_path or cv_dir must be provided")

    print(f"Using {model_type} model for predictions")

    # Load JSON data
    print(f"\nLoading JSON from: {json_path}")
    json_data = load_json_data(json_path)

    # Get file keys to process
    if file_keys is None:
        file_keys = list(json_data.keys())
    import random

    random.shuffle(file_keys)

    if max_samples:
        file_keys = file_keys[:max_samples]

    print(f"Visualizing {len(file_keys)} samples...")
    print("-" * 60)

    all_metrics = {"mae": [], "accuracy": [], "precision": [], "recall": []}

    for file_key in file_keys:
        save_path = os.path.join(output_dir, f"{file_key}_pred.png")

        try:
            builder, predictions, ground_truth = predict_single_sample(json_data, file_key, model, device)

            if builder is None:
                print(f"Skipped: {file_key} (failed to build graph)")
                continue

            metrics = visualize_prediction(builder, predictions, ground_truth, file_key, save_path)

            for k, v in metrics.items():
                all_metrics[k].append(v)

        except Exception as e:
            print(f"Error processing {file_key}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print(f"Total samples: {len(all_metrics['mae'])}")
    for k, v in all_metrics.items():
        if v:
            print(f"  {k.upper():12s}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")

    return all_metrics


def get_file_keys_from_dir(dir_path: str) -> List[str]:
    """Get file keys from a directory containing DXF files."""
    import os

    file_keys = []
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dxf"):
                key = os.path.splitext(fname)[0]
                file_keys.append(key)
    return file_keys


def main():
    parser = argparse.ArgumentParser(description="Visualize Edge GNN predictions")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (for single model)",
    )
    parser.add_argument(
        "--cv_dir",
        type=str,
        default=None,
        help="Directory containing fold models (for ensemble)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of folds for ensemble (auto-detect if not specified)",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=data_config.JSON_PATH,
        help="Path to JSON data file",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=data_config.TEST_DIR,
        help="Directory containing test DXF files (use filenames as keys)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/result/shearwall_pred/baseline-GNN-EP4/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to visualize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Check that either model_path or cv_dir is provided
    if args.model_path is None and args.cv_dir is None:
        # Default to single model path
        args.model_path = train_config.MODEL_SAVE_PATH
        print(f"No model specified, using default: {args.model_path}")

    # Get file keys from test directory
    test_keys = get_file_keys_from_dir(args.test_dir)
    print(f"Found {len(test_keys)} test files in {args.test_dir}")

    run_test_visualization(
        model_path=args.model_path,
        cv_dir=args.cv_dir,
        n_folds=args.n_folds,
        json_path=args.json_path,
        output_dir=args.output_dir,
        file_keys=test_keys,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
