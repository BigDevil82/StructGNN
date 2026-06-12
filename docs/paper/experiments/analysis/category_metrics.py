"""
按设计类别统计评估指标

读取Ours和Baseline的per_sample指标文件，按照三种设计类别分别统计各项指标的平均值：
- L17: 低剪力墙比例
- L27: 中剪力墙比例
- L1L28: 高剪力墙比例
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np


def get_file_category(file_key: str) -> int:
    """
    根据文件名获取设计类别

    Categories:
    - 0: L17 (低剪力墙比例)
    - 1: L27 (中剪力墙比例)
    - 2: L1L28 (高剪力墙比例)

    Returns:
        Category index (0, 1, 2) or -1 if unknown
    """
    if file_key.startswith("L17"):
        return 0
    elif file_key.startswith("L27"):
        return 1
    elif file_key.startswith("L1L28"):
        return 2
    return -1


def get_category_name(category: int) -> str:
    """获取类别名称"""
    category_names = {
        0: "L17 (Low)",
        1: "L27 (Medium)",
        2: "L1L28 (High)",
    }
    return category_names.get(category, "Unknown")


def load_and_categorize_metrics(json_path: str) -> Dict[int, List[Dict]]:
    """
    加载指标文件并按类别分组

    Returns:
        {category_id: [samples]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categorized = defaultdict(list)

    for sample in data:
        file_key = sample["file_key"]
        category = get_file_category(file_key)

        if category >= 0:
            categorized[category].append(sample)

    return categorized


def compute_category_statistics(categorized_data: Dict[int, List[Dict]]) -> Dict:
    """
    计算每个类别的指标统计值

    Returns:
        {category_id: {metric_name: {mean, std, ...}}}
    """
    statistics = {}

    # 关注的指标
    metrics_of_interest = ["image_iou", "precision", "recall", "f1", "accuracy", "mae", "rmse"]

    for category, samples in categorized_data.items():
        if not samples:
            continue

        category_stats = {}

        for metric in metrics_of_interest:
            values = [s[metric] for s in samples if metric in s]

            if values:
                category_stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

        statistics[category] = category_stats

    return statistics


def print_comparison_table(ours_stats: Dict, baseline_stats: Dict):
    """打印对比表格"""

    print("\n" + "=" * 120)
    print("METRICS BY DESIGN CATEGORY")
    print("=" * 120)

    # 关注的主要指标
    main_metrics = ["image_iou", "precision", "recall", "f1", "mae"]

    for metric in main_metrics:
        print(f"\n{'─' * 120}")
        print(f"Metric: {metric.upper()}")
        print(f"{'─' * 120}")
        print(
            f"{'Category':<20} {'Ours Mean':<15} {'Ours Std':<15} {'Baseline Mean':<15} {'Baseline Std':<15} {'Improvement':<15}"
        )
        print(f"{'─' * 120}")

        for category in sorted(ours_stats.keys()):
            category_name = get_category_name(category)

            ours_mean = ours_stats[category][metric]["mean"]
            ours_std = ours_stats[category][metric]["std"]
            baseline_mean = baseline_stats[category][metric]["mean"]
            baseline_std = baseline_stats[category][metric]["std"]

            # 计算改进（对于MAE，越小越好；其他指标越大越好）
            if metric == "mae":
                improvement = ((baseline_mean - ours_mean) / baseline_mean) * 100
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement = ((ours_mean - baseline_mean) / baseline_mean) * 100
                improvement_str = f"{improvement:+.2f}%"

            print(
                f"{category_name:<20} {ours_mean:<15.4f} {ours_std:<15.4f} {baseline_mean:<15.4f} {baseline_std:<15.4f} {improvement_str:<15}"
            )

        # 打印总体平均
        print(f"{'─' * 120}")
        all_ours_means = [ours_stats[c][metric]["mean"] for c in ours_stats if metric in ours_stats[c]]
        ours_std_mean = [ours_stats[c][metric]["std"] for c in ours_stats if metric in ours_stats[c]]
        all_baseline_means = [
            baseline_stats[c][metric]["mean"] for c in baseline_stats if metric in baseline_stats[c]
        ]
        baseline_std_mean = [
            baseline_stats[c][metric]["std"] for c in baseline_stats if metric in baseline_stats[c]
        ]

        if all_ours_means and all_baseline_means:
            avg_ours = np.mean(all_ours_means)
            avg_baseline = np.mean(all_baseline_means)

            if metric == "mae":
                avg_improvement = ((avg_baseline - avg_ours) / avg_baseline) * 100
            else:
                avg_improvement = ((avg_ours - avg_baseline) / avg_baseline) * 100

            print(
                f"{'Overall Average':<20} {avg_ours:<15.4f} {np.mean(ours_std_mean):<15.4f} {avg_baseline:<15.4f} {np.mean(baseline_std_mean):<15.4f} {avg_improvement:+.2f}%"
            )

    print("\n" + "=" * 120)


def save_category_statistics(
    ours_stats: Dict,
    baseline_stats: Dict,
    output_file: str,
):
    """保存类别统计结果到JSON"""

    output_data = {
        "ours": {get_category_name(cat): stats for cat, stats in ours_stats.items()},
        "baseline": {get_category_name(cat): stats for cat, stats in baseline_stats.items()},
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed statistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze metrics by design category")
    parser.add_argument(
        "--ours_metrics",
        type=str,
        default="outputs/result/metrics/image_iou_comparison/ours_per_sample.json",
        help="Ours model per-sample metrics JSON",
    )
    parser.add_argument(
        "--baseline_metrics",
        type=str,
        default="outputs/result/metrics/image_iou_comparison/baseline_per_sample.json",
        help="Baseline model per-sample metrics JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result/metrics/category_statistics.json",
        help="Output JSON file for category statistics",
    )

    args = parser.parse_args()

    print("Loading metrics files...")

    # 加载并分类
    ours_categorized = load_and_categorize_metrics(args.ours_metrics)
    baseline_categorized = load_and_categorize_metrics(args.baseline_metrics)

    print(f"Ours samples by category:")
    for cat, samples in sorted(ours_categorized.items()):
        print(f"  {get_category_name(cat)}: {len(samples)} samples")

    print(f"\nBaseline samples by category:")
    for cat, samples in sorted(baseline_categorized.items()):
        print(f"  {get_category_name(cat)}: {len(samples)} samples")

    # 计算统计值
    print("\nComputing statistics...")
    ours_stats = compute_category_statistics(ours_categorized)
    baseline_stats = compute_category_statistics(baseline_categorized)

    # 打印对比表格
    print_comparison_table(ours_stats, baseline_stats)

    # 保存结果
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_category_statistics(ours_stats, baseline_stats, args.output)


if __name__ == "__main__":
    main()
