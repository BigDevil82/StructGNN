"""
图表征统计脚本

计算Ours（房间级图）和Baseline（边级图）两种表征方式的图统计信息：
- 平均节点数
- 平均边数
- 节点数/边数分布
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.research.baseline_edge_gnn.config import data_config
from experiments.research.baseline_edge_gnn.graph_builder import build_graph_from_json, load_json_data
from shearwall_pred.utils import build_graph_from_dxf


def compute_ours_graph_stats(dxf_path: str) -> Tuple[int, int]:
    """
    计算Ours方法的图统计信息

    Returns:
        (num_nodes, num_edges): 节点数和边数
    """
    try:
        builder = build_graph_from_dxf(dxf_path, mode="none")
        graph = builder.graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        return num_nodes, num_edges
    except Exception as e:
        print(f"Error processing {dxf_path}: {e}")
        return None, None


def compute_baseline_graph_stats(json_data: Dict, file_key: str) -> Tuple[int, int]:
    """
    计算Baseline方法的图统计信息

    Returns:
        (num_nodes, num_edges): 节点数（边端点）和边数（边之间的连接）
    """
    try:
        builder = build_graph_from_json(json_data, file_key, mode="none")
        if builder is None:
            return None, None

        data = builder.to_pyg_data()
        if data is None:
            return None, None

        # Baseline的图结构：节点是边的端点，边是端点之间的连接
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]  # 边的数量（包括双向）

        return num_nodes, num_edges
    except Exception as e:
        print(f"Error processing {file_key}: {e}")
        return None, None


def analyze_graph_statistics(
    test_dir: str,
    json_path: str,
    output_file: str = None,
):
    """
    分析测试集上两种表征方式的图统计信息
    """
    # 加载JSON数据（用于baseline）
    print("Loading JSON data...")
    json_data = load_json_data(json_path)

    # 获取测试集文件列表
    test_files = []
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(".dxf"):
            file_key = os.path.splitext(fname)[0]
            dxf_path = os.path.join(test_dir, fname)
            test_files.append((file_key, dxf_path))

    print(f"Found {len(test_files)} test files")
    print("-" * 80)

    # 统计信息
    ours_stats = {"nodes": [], "edges": []}
    baseline_stats = {"nodes": [], "edges": []}

    per_sample_stats = []

    # 处理每个测试样本
    for file_key, dxf_path in test_files:
        print(f"Processing: {file_key}")

        # Ours统计
        ours_nodes, ours_edges = compute_ours_graph_stats(dxf_path)

        # Baseline统计
        baseline_nodes, baseline_edges = compute_baseline_graph_stats(json_data, file_key)

        if ours_nodes is not None and baseline_nodes is not None:
            ours_stats["nodes"].append(ours_nodes)
            ours_stats["edges"].append(ours_edges)
            baseline_stats["nodes"].append(baseline_nodes)
            baseline_stats["edges"].append(baseline_edges)

            per_sample_stats.append(
                {
                    "file_key": file_key,
                    "ours_nodes": ours_nodes,
                    "ours_edges": ours_edges,
                    "baseline_nodes": baseline_nodes,
                    "baseline_edges": baseline_edges,
                }
            )

            print(f"  Ours: {ours_nodes} nodes, {ours_edges} edges")
            print(f"  Baseline: {baseline_nodes} nodes, {baseline_edges} edges")
        else:
            print(f"  Skipped (failed to build graph)")

    # 计算统计量
    print("\n" + "=" * 80)
    print("GRAPH STATISTICS SUMMARY")
    print("=" * 80)

    print(f"\nTotal samples analyzed: {len(ours_stats['nodes'])}")

    print("\n--- Ours (Room-level Graph) ---")
    print(f"Nodes:")
    print(f"  Mean:   {np.mean(ours_stats['nodes']):.2f}")
    print(f"  Std:    {np.std(ours_stats['nodes']):.2f}")
    print(f"  Median: {np.median(ours_stats['nodes']):.0f}")
    print(f"  Min:    {np.min(ours_stats['nodes'])}")
    print(f"  Max:    {np.max(ours_stats['nodes'])}")

    print(f"\nEdges:")
    print(f"  Mean:   {np.mean(ours_stats['edges']):.2f}")
    print(f"  Std:    {np.std(ours_stats['edges']):.2f}")
    print(f"  Median: {np.median(ours_stats['edges']):.0f}")
    print(f"  Min:    {np.min(ours_stats['edges'])}")
    print(f"  Max:    {np.max(ours_stats['edges'])}")

    print("\n--- Baseline (Edge-level Graph) ---")
    print(f"Nodes:")
    print(f"  Mean:   {np.mean(baseline_stats['nodes']):.2f}")
    print(f"  Std:    {np.std(baseline_stats['nodes']):.2f}")
    print(f"  Median: {np.median(baseline_stats['nodes']):.0f}")
    print(f"  Min:    {np.min(baseline_stats['nodes'])}")
    print(f"  Max:    {np.max(baseline_stats['nodes'])}")

    print(f"\nEdges:")
    print(f"  Mean:   {np.mean(baseline_stats['edges']):.2f}")
    print(f"  Std:    {np.std(baseline_stats['edges']):.2f}")
    print(f"  Median: {np.median(baseline_stats['edges']):.0f}")
    print(f"  Min:    {np.min(baseline_stats['edges'])}")
    print(f"  Max:    {np.max(baseline_stats['edges'])}")

    # 对比
    print("\n--- Comparison (Baseline / Ours) ---")
    node_ratio = np.mean(baseline_stats["nodes"]) / np.mean(ours_stats["nodes"])
    edge_ratio = np.mean(baseline_stats["edges"]) / np.mean(ours_stats["edges"])
    print(f"Avg nodes ratio:  {node_ratio:.2f}x")
    print(f"Avg edges ratio:  {edge_ratio:.2f}x")

    print("=" * 80)

    # 保存详细统计到JSON
    if output_file:
        output_data = {
            "summary": {
                "total_samples": len(ours_stats["nodes"]),
                "ours": {
                    "nodes": {
                        "mean": float(np.mean(ours_stats["nodes"])),
                        "std": float(np.std(ours_stats["nodes"])),
                        "median": float(np.median(ours_stats["nodes"])),
                        "min": int(np.min(ours_stats["nodes"])),
                        "max": int(np.max(ours_stats["nodes"])),
                    },
                    "edges": {
                        "mean": float(np.mean(ours_stats["edges"])),
                        "std": float(np.std(ours_stats["edges"])),
                        "median": float(np.median(ours_stats["edges"])),
                        "min": int(np.min(ours_stats["edges"])),
                        "max": int(np.max(ours_stats["edges"])),
                    },
                },
                "baseline": {
                    "nodes": {
                        "mean": float(np.mean(baseline_stats["nodes"])),
                        "std": float(np.std(baseline_stats["nodes"])),
                        "median": float(np.median(baseline_stats["nodes"])),
                        "min": int(np.min(baseline_stats["nodes"])),
                        "max": int(np.max(baseline_stats["nodes"])),
                    },
                    "edges": {
                        "mean": float(np.mean(baseline_stats["edges"])),
                        "std": float(np.std(baseline_stats["edges"])),
                        "median": float(np.median(baseline_stats["edges"])),
                        "min": int(np.min(baseline_stats["edges"])),
                        "max": int(np.max(baseline_stats["edges"])),
                    },
                },
                "comparison": {
                    "node_ratio": float(node_ratio),
                    "edge_ratio": float(edge_ratio),
                },
            },
            "per_sample": per_sample_stats,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed statistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze graph statistics for Ours vs Baseline")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/dxf/shearwall_split_8_2/train",
        help="Test DXF files directory",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=data_config.JSON_PATH,
        help="JSON data file for baseline",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result/metrics/graph_statistics.json",
        help="Output JSON file for detailed statistics",
    )

    args = parser.parse_args()

    analyze_graph_statistics(
        test_dir=args.test_dir,
        json_path=args.json_path,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
