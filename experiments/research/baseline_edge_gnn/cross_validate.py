"""
Cross-validation for Edge-based GNN Baseline

Performs K-fold cross-validation with stratified splits by file.
Uses file keys from train/test directories combined.
"""

import argparse
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.research.baseline_edge_gnn.config import data_config, model_config, train_config
from experiments.research.baseline_edge_gnn.dataset import EdgeShearWallDataset
from experiments.research.baseline_edge_gnn.graph_builder import get_file_category
from experiments.research.baseline_edge_gnn.model import GNNEP4
from experiments.research.baseline_edge_gnn.trainer import EdgeGNNTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_file_keys_from_dir(dir_path: str) -> List[str]:
    """Get file keys from a directory containing DXF files."""
    file_keys = []
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(".dxf"):
                key = os.path.splitext(fname)[0]
                file_keys.append(key)
    return file_keys


def cross_validate(
    json_path: str,
    train_file_keys: List[str],
    test_file_keys: List[str],
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 8,
    seed: int = 42,
    save_dir: str = "models/baseline_cv",
    cache_dir: str = "data/cache/baseline_cv",
) -> Dict:
    """
    Perform K-fold cross-validation.

    Args:
        json_path: Path to JSON data file
        train_file_keys: List of training file keys for CV split
        test_file_keys: List of test file keys (fixed test set)
        n_folds: Number of folds
        epochs: Training epochs per fold
        batch_size: Batch size
        seed: Random seed
        save_dir: Directory to save fold models
        cache_dir: Cache directory for processed data

    Returns:
        Dictionary with aggregated metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get category for each training file (for stratified split)
    train_categories = [get_file_category(key) for key in train_file_keys]

    # Stratified K-fold on training files (by category)
    skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_results = []
    fold_histories = []

    print("=" * 60)
    print(f"Edge-based GNN Cross-Validation ({n_folds} folds)")
    print(f"Training files for CV: {len(train_file_keys)}")
    print(f"Fixed test files: {len(test_file_keys)}")
    print("=" * 60)

    # Count category distribution
    from collections import Counter

    train_cat_counts = Counter(train_categories)
    print(f"Train category distribution: {dict(train_cat_counts)}")

    for fold, (train_idx, val_idx) in enumerate(skfold.split(train_file_keys, train_categories)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("=" * 60)

        set_seed(seed + fold)

        # Get file keys for this fold (train/val from CV split, test is fixed)
        fold_train_keys = [train_file_keys[i] for i in train_idx]
        fold_val_keys = [train_file_keys[i] for i in val_idx]

        print(
            f"Train files: {len(fold_train_keys)}, Val files: {len(fold_val_keys)}, Test files: {len(test_file_keys)}"
        )

        # Create datasets for this fold
        fold_cache_dir = os.path.join(cache_dir, f"fold_{fold + 1}")

        train_dataset = EdgeShearWallDataset(
            root=os.path.join(fold_cache_dir, "train"),
            json_path=json_path,
            file_keys=fold_train_keys,
            is_test=False,
        )
        val_dataset = EdgeShearWallDataset(
            root=os.path.join(fold_cache_dir, "val"),
            json_path=json_path,
            file_keys=fold_val_keys,
            is_test=True,
        )
        test_dataset = EdgeShearWallDataset(
            root=os.path.join(fold_cache_dir, "test"),
            json_path=json_path,
            file_keys=test_file_keys,
            is_test=True,
        )

        print(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
        )

        # Create weighted sampler for balanced training
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Create model
        model = GNNEP4(
            node_in_dim=model_config.NODE_IN_DIM,
            edge_in_dim=model_config.EDGE_IN_DIM,
            gnn_out_dim=model_config.GNN_OUT_DIM,
            dropout=model_config.DROPOUT,
        )

        # Train
        trainer = EdgeGNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        save_path = os.path.join(save_dir, f"fold_{fold + 1}.pt")
        history = trainer.train(
            epochs=epochs,
            save_path=save_path,
            verbose=True,
        )

        # Evaluate
        results = trainer.evaluate()
        all_results.append(results)
        fold_histories.append(history)

    # Aggregate results
    print("\n" + "=" * 60)
    print("Cross-Validation Results Summary")
    print("=" * 60)

    metrics = ["loss", "mae", "accuracy", "precision", "recall"]
    summary = {}

    for metric in metrics:
        values = [r.get(metric, 0) for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric] = {"mean": mean_val, "std": std_val, "values": values}
        print(f"{metric.upper():12s}: {mean_val:.4f} +/- {std_val:.4f}")

    print("=" * 60)

    # Save summary
    summary_path = os.path.join(save_dir, "cv_summary.pt")
    torch.save(
        {
            "outputs/result": all_results,
            "summary": summary,
            "histories": fold_histories,
            "n_folds": n_folds,
        },
        summary_path,
    )
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-validate Edge-based GNN")
    parser.add_argument(
        "--json_path",
        type=str,
        default=data_config.JSON_PATH,
        help="Path to JSON data file",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=data_config.TRAIN_DIR,
        help="Directory containing training DXF files",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=data_config.TEST_DIR,
        help="Directory containing test DXF files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=data_config.CACHE_DIR,
        help="Cache directory for processed data",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.EPOCHS,
        help="Epochs per fold",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=train_config.BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=train_config.SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/result/shearwall_pred/baseline-GNN-EP4-CV",
        help="Directory to save models",
    )

    args = parser.parse_args()

    # Get file keys from train and test directories separately
    train_keys = get_file_keys_from_dir(args.train_dir)
    test_keys = get_file_keys_from_dir(args.test_dir)

    print(f"Train directory: {args.train_dir} ({len(train_keys)} files)")
    print(f"Test directory: {args.test_dir} ({len(test_keys)} files)")

    # Run cross-validation (only split train files, test is fixed)
    summary = cross_validate(
        json_path=args.json_path,
        train_file_keys=train_keys,
        test_file_keys=test_keys,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        save_dir=args.save_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
