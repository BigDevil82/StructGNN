"""
Analyze category bias in training data.

Check if model is learning category-specific patterns instead of structural features.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.research.baseline_edge_gnn.config import data_config
from experiments.research.baseline_edge_gnn.graph_builder import build_graph_from_json, get_file_category, load_json_data


def analyze_category_distribution(json_path: str, train_dir: str, test_dir: str):
    """
    Analyze shear wall ratio distribution across categories.
    """
    # Load JSON data
    print(f"Loading JSON from: {json_path}")
    json_data = load_json_data(json_path)

    # Get train and test file keys
    def get_keys(dir_path):
        keys = []
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.lower().endswith('.dxf'):
                    keys.append(os.path.splitext(fname)[0])
        return keys

    train_keys = get_keys(train_dir)
    test_keys = get_keys(test_dir)

    print(f"\nTrain files: {len(train_keys)}")
    print(f"Test files: {len(test_keys)}")

    # Analyze each split
    for split_name, file_keys in [("Train", train_keys), ("Test", test_keys)]:
        print(f"\n{'='*60}")
        print(f"{split_name} Set Analysis")
        print('='*60)

        category_stats = defaultdict(lambda: {"ratios": [], "file_count": 0})

        for file_key in file_keys:
            category = get_file_category(file_key)
            if category == -1:
                continue

            # Build graph to get ground truth
            builder = build_graph_from_json(json_data, file_key, mode="none")
            if builder is None:
                continue

            data = builder.to_pyg_data()
            if data is None:
                continue

            # Get ground truth ratios (only for PSW edges)
            gt = data.y.cpu().numpy()  # [num_edges * 2, 2]

            # Flatten and collect non-zero ratios
            ratios = gt.flatten()
            ratios = ratios[ratios > 0.01]  # Filter out near-zero values

            category_stats[category]["ratios"].extend(ratios.tolist())
            category_stats[category]["file_count"] += 1

        # Print statistics
        category_names = {0: "L17 (低层)", 1: "L27 (中高层)", 2: "L1L28 (超高层)"}

        print(f"\n{'Category':<20} {'Files':<10} {'Samples':<10} {'Mean':<10} {'Median':<10} {'Std':<10}")
        print("-" * 70)

        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            ratios = np.array(stats["ratios"])

            if len(ratios) > 0:
                print(
                    f"{category_names.get(cat, f'Cat{cat}'):<20} "
                    f"{stats['file_count']:<10} "
                    f"{len(ratios):<10} "
                    f"{ratios.mean():.4f}    "
                    f"{np.median(ratios):.4f}    "
                    f"{ratios.std():.4f}"
                )

        # Plot distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, cat in enumerate(sorted(category_stats.keys())):
            stats = category_stats[cat]
            ratios = np.array(stats["ratios"])

            if len(ratios) > 0:
                axes[idx].hist(ratios, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].axvline(ratios.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratios.mean():.3f}')
                axes[idx].set_xlabel('Shear Wall Ratio')
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{category_names.get(cat, f"Cat{cat}")} ({stats["file_count"]} files)')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f'{split_name} Set: Shear Wall Ratio Distribution by Category')
        plt.tight_layout()

        output_path = f"category_distribution_{split_name.lower()}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved distribution plot to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze category bias")
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
        help="Train directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=data_config.TEST_DIR,
        help="Test directory",
    )

    args = parser.parse_args()

    analyze_category_distribution(args.json_path, args.train_dir, args.test_dir)


if __name__ == "__main__":
    main()
