"""
Training script for Edge-based GNN Baseline (JSON Format)

Usage:
    python -m baseline_edge_gnn.train --json_path data/edge_graph_data.json
"""

import argparse
import os
import random
import sys
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.research.baseline_edge_gnn.config import data_config, model_config, train_config
from experiments.research.baseline_edge_gnn.dataset import EdgeShearWallDataset
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
    """
    Get file keys from a directory containing DXF files.

    Returns file names without extension as keys.
    """
    file_keys = []
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.lower().endswith('.dxf'):
                # Remove .dxf extension to get the key
                key = os.path.splitext(fname)[0]
                file_keys.append(key)
    return file_keys


def main():
    parser = argparse.ArgumentParser(description="Train Edge-based GNN Baseline")
    parser.add_argument(
        "--json_path",
        type=str,
        default=data_config.JSON_PATH,
        help="Path to JSON file containing all samples",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=data_config.TRAIN_DIR,
        help="Directory containing training DXF files (use filenames as keys)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=data_config.TEST_DIR,
        help="Directory containing test DXF files (use filenames as keys)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=data_config.CACHE_DIR,
        help="Cache directory for processed data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.EPOCHS,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=train_config.BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=train_config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=train_config.SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio from training set",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=train_config.MODEL_SAVE_PATH,
        help="Path to save trained model",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print("=" * 60)
    print("Edge-based GNN Baseline Training (JSON Format)")
    print("=" * 60)

    # Get file keys from train/test directories
    train_keys = get_file_keys_from_dir(args.train_dir)
    test_keys = get_file_keys_from_dir(args.test_dir)

    print(f"\nTrain directory: {args.train_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Train files: {len(train_keys)}, Test files: {len(test_keys)}")

    # Split train into train/val
    if args.val_ratio > 0:
        train_keys, val_keys = train_test_split(
            train_keys, test_size=args.val_ratio, random_state=args.seed
        )
    else:
        val_keys = []

    print(f"After val split - Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")

    # Load datasets
    print(f"\nLoading dataset from: {args.json_path}")
    print(f"Cache directory: {args.cache_dir}")

    train_dataset = EdgeShearWallDataset(
        root=os.path.join(args.cache_dir, "train"),
        json_path=args.json_path,
        file_keys=train_keys,
    )
    val_dataset = EdgeShearWallDataset(
        root=os.path.join(args.cache_dir, "val"),
        json_path=args.json_path,
        file_keys=val_keys,
        is_test=True,  # No augmentation for validation
    ) if val_keys else None
    test_dataset = EdgeShearWallDataset(
        root=os.path.join(args.cache_dir, "test"),
        json_path=args.json_path,
        file_keys=test_keys,
        is_test=True,  # No augmentation for test
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset) if val_dataset else 0}, Test samples: {len(test_dataset)}")

    # Create weighted sampler for balanced training
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nInitializing GNN-EP-4 model...")
    model = GNNEP4(
        node_in_dim=model_config.NODE_IN_DIM,
        edge_in_dim=model_config.EDGE_IN_DIM,
        gnn_out_dim=model_config.GNN_OUT_DIM,
        dropout=model_config.DROPOUT,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = EdgeGNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.lr,
    )

    # Train
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    history = trainer.train(
        epochs=args.epochs,
        save_path=args.save_path,
        verbose=True,
    )

    # Evaluate
    trainer.evaluate()

    # Plot history
    history_path = args.save_path.replace(".pt", "_history.png")
    trainer.plot_history(save_path=history_path)

    print("\nTraining completed!")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
