"""
对比表格：房间节点 vs 几何图元的差异分析

用于论文的Related Work或Methodology章节
"""

import pandas as pd
from pathlib import Path


def create_comparison_table(output_path: Path = None):
    """
    创建房间节点与几何图元方法的系统对比表格
    """
    comparison_data = {
        'Aspect': [
            'Graph Construction',
            'Node Semantics',
            'Feature Richness',
            'Structural Rationality',
            'Design Code Alignment',
            'Extensibility to Beams/Columns',
            'Interpretability',
            'Computational Complexity',
            'Conditional Generation',
            'Handling Irregular Layouts'
        ],
        'Geometric Primitive-based': [
            'Wall segments as nodes',
            'Low-level geometry (line, point)',
            'Limited (length, orientation, coordinates)',
            'Local analysis of isolated segments',
            'Indirect - requires post-processing',
            'Difficult - different granularity',
            'Black-box predictions',
            'High (many fine-grained nodes)',
            'Not explicitly supported',
            'Generates fragmented predictions'
        ],
        'Room-based (Ours)': [
            'Rooms as nodes',
            'High-level architectural space',
            'Rich (area, aspect ratio, function, adjacency)',
            'Holistic load transfer modeling',
            'Direct - room is the design unit in codes',
            'Natural - beams along edges, columns at corners',
            'Predictions align with design logic',
            'Lower (fewer semantic nodes)',
            'Inherent via FiLM conditioning',
            'Maintains spatial coherence'
        ],
        'Advantage': [
            '✓',
            '✓✓',
            '✓✓',
            '✓✓✓',
            '✓✓',
            '✓✓✓',
            '✓✓',
            '✓',
            '✓✓',
            '✓✓'
        ]
    }

    df = pd.DataFrame(comparison_data)

    # 保存为LaTeX表格
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV格式
        df.to_csv(output_path.with_suffix('.csv'), index=False)

        # LaTeX格式
        latex_table = df.to_latex(
            index=False,
            column_format='p{4cm}|p{5cm}|p{5cm}|c',
            caption='Comparison between geometric primitive-based and room-based graph representations',
            label='tab:method_comparison',
            escape=False
        )

        # 美化LaTeX表格
        latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
        latex_table = latex_table.replace('\\midrule', '\\hline')
        latex_table = latex_table.replace('\\bottomrule', '\\hline\\hline')

        with open(output_path.with_suffix('.tex'), 'w', encoding='utf-8') as f:
            f.write(latex_table)

        # Markdown格式（用于README或初稿）
        with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=False))

        print(f"Comparison table saved to:")
        print(f"  - CSV: {output_path.with_suffix('.csv')}")
        print(f"  - LaTeX: {output_path.with_suffix('.tex')}")
        print(f"  - Markdown: {output_path.with_suffix('.md')}")

    return df


def create_quantitative_comparison_table(
    baseline_results: dict,
    our_results: dict,
    output_path: Path = None
):
    """
    创建定量对比表格（如果你已经有baseline实验数据）

    Args:
        baseline_results: 几何图元方法的结果 {'iou': 0.58, 'f1': 0.72, 'mae': 0.15}
        our_results: 房间节点方法的结果 {'iou': 0.65, 'f1': 0.85, 'mae': 0.08}
    """
    metrics = ['Vector IoU', 'F1 Score', 'MAE', 'Precision', 'Recall']

    baseline_values = [
        baseline_results.get('iou', 0.0),
        baseline_results.get('f1', 0.0),
        baseline_results.get('mae', 0.0),
        baseline_results.get('precision', 0.0),
        baseline_results.get('recall', 0.0)
    ]

    our_values = [
        our_results.get('iou', 0.0),
        our_results.get('f1', 0.0),
        our_results.get('mae', 0.0),
        our_results.get('precision', 0.0),
        our_results.get('recall', 0.0)
    ]

    improvements = [
        ((our_values[i] - baseline_values[i]) / baseline_values[i] * 100)
        if baseline_values[i] > 0 else 0
        for i in range(len(metrics))
    ]

    # 对于MAE，改进应该是负数（越小越好）
    improvements[2] = -improvements[2]

    comparison_data = {
        'Metric': metrics,
        'Geometric Primitive-based': [f'{v:.4f}' for v in baseline_values],
        'Room-based (Ours)': [f'{v:.4f}' for v in our_values],
        'Improvement': [f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%' for imp in improvements]
    }

    df = pd.DataFrame(comparison_data)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path.with_suffix('.csv'), index=False)

        latex_table = df.to_latex(
            index=False,
            column_format='l|c|c|c',
            caption='Quantitative comparison on test set',
            label='tab:quantitative_comparison',
            escape=False
        )

        with open(output_path.with_suffix('.tex'), 'w', encoding='utf-8') as f:
            f.write(latex_table)

        with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=False))

        print(f"\nQuantitative comparison table saved to:")
        print(f"  - CSV: {output_path.with_suffix('.csv')}")
        print(f"  - LaTeX: {output_path.with_suffix('.tex')}")
        print(f"  - Markdown: {output_path.with_suffix('.md')}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create comparison tables for paper")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/analysis/tables",
                        help="Output directory for tables")
    parser.add_argument("--baseline_results", type=str, default=None,
                        help="Path to baseline results JSON file")
    parser.add_argument("--our_results", type=str, default=None,
                        help="Path to our results JSON file")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建定性对比表
    print("Creating qualitative comparison table...")
    create_comparison_table(output_dir / "qualitative_comparison")

    # 如果提供了结果文件，创建定量对比表
    if args.baseline_results and args.our_results:
        import json

        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)

        with open(args.our_results, 'r') as f:
            our_results = json.load(f)

        print("\nCreating quantitative comparison table...")
        create_quantitative_comparison_table(
            baseline_results,
            our_results,
            output_dir / "quantitative_comparison"
        )

    print("\n✓ All tables created successfully!")
