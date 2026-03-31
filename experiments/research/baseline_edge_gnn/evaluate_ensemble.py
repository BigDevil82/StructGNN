"""
Evaluate ensemble model from K-fold cross-validation.

Uses averaged predictions from multiple fold models.
"""

import argparse
import os
import sys

from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.research.baseline_edge_gnn.config import data_config, model_config, train_config
from experiments.research.baseline_edge_gnn.dataset import EdgeShearWallDataset
from experiments.research.baseline_edge_gnn.ensemble_model import EnsembleGNNEP4
from experiments.research.baseline_edge_gnn.trainer import EdgeGNNTrainer


def get_file_keys_from_dir(dir_path: str):
    """Get file keys from a directory containing DXF files."""
    file_keys = []
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dxf"):
                key = os.path.splitext(fname)[0]
                file_keys.append(key)
    return file_keys


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble Edge-based GNN")
    parser.add_argument(
        "--cv_dir",
        type=str,
        default="outputs/result/shearwall_pred/baseline-GNN-EP4/cv_models",
        help="Directory containing fold models",
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
        help="Test directory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache/baseline_ensemble_test",
        help="Cache directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=train_config.BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of folds (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Get test file keys
    test_keys = get_file_keys_from_dir(args.test_dir)
    print(f"Test directory: {args.test_dir} ({len(test_keys)} files)")

    # Create test dataset
    test_dataset = EdgeShearWallDataset(
        root=args.cache_dir,
        json_path=args.json_path,
        file_keys=test_keys,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Load ensemble model
    print("\nLoading ensemble model...")
    ensemble_model = EnsembleGNNEP4.from_cv_dir(
        cv_dir=args.cv_dir,
        node_in_dim=model_config.NODE_IN_DIM,
        edge_in_dim=model_config.EDGE_IN_DIM,
        gnn_out_dim=model_config.GNN_OUT_DIM,
        dropout=model_config.DROPOUT,
        n_folds=args.n_folds,
    )

    # Create trainer with ensemble model
    trainer = EdgeGNNTrainer(
        model=ensemble_model,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
    )

    # Evaluate
    print("\nEvaluating ensemble model on test set...")
    print("=" * 60)
    results = trainer.evaluate()

    print("\nEnsemble Test Results:")
    print("=" * 60)
    print(f"MAE:       {results['mae']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results.get('f1', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
