"""
统计检验：房间级预测的优越性

通过配对t检验等方法验证房间节点方法在统计上显著优于几何图元方法
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def paired_t_test(
    baseline_metrics: np.ndarray,
    our_metrics: np.ndarray,
    metric_name: str = "IoU"
):
    """
    配对t检验：逐样本比较两种方法的性能

    Args:
        baseline_metrics: 基线方法在每个测试样本上的指标 (N,)
        our_metrics: 我们的方法在每个测试样本上的指标 (N,)
        metric_name: 指标名称

    Returns:
        dict: 统计检验结果
    """
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(our_metrics, baseline_metrics)

    # 效应量（Cohen's d）
    diff = our_metrics - baseline_metrics
    cohens_d = diff.mean() / diff.std()

    # 置信区间
    ci = stats.t.interval(
        0.95,
        len(diff) - 1,
        loc=diff.mean(),
        scale=stats.sem(diff)
    )

    result = {
        'metric': metric_name,
        'baseline_mean': float(baseline_metrics.mean()),
        'baseline_std': float(baseline_metrics.std()),
        'our_mean': float(our_metrics.mean()),
        'our_std': float(our_metrics.std()),
        'improvement': float(diff.mean()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        '95_ci_lower': float(ci[0]),
        '95_ci_upper': float(ci[1]),
        'significant': p_value < 0.05
    }

    return result


def wilcoxon_test(
    baseline_metrics: np.ndarray,
    our_metrics: np.ndarray,
    metric_name: str = "IoU"
):
    """
    Wilcoxon符号秩检验（非参数检验，当数据不满足正态分布时使用）
    """
    stat, p_value = stats.wilcoxon(our_metrics, baseline_metrics)

    result = {
        'metric': metric_name,
        'baseline_median': float(np.median(baseline_metrics)),
        'our_median': float(np.median(our_metrics)),
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }

    return result


def visualize_comparison(
    baseline_metrics: dict,
    our_metrics: dict,
    output_path: Path
):
    """
    可视化对比：箱线图 + 显著性标注

    Args:
        baseline_metrics: {'iou': [0.5, 0.6, ...], 'f1': [...], ...}
        our_metrics: {'iou': [0.6, 0.7, ...], 'f1': [...], ...}
    """
    metrics = list(baseline_metrics.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        baseline_data = np.array(baseline_metrics[metric])
        our_data = np.array(our_metrics[metric])

        # 箱线图
        data_to_plot = [baseline_data, our_data]
        bp = ax.boxplot(data_to_plot, labels=['Geometric\nPrimitive', 'Room-based\n(Ours)'],
                        patch_artist=True, widths=0.6)

        # 美化
        colors = ['#FF9999', '#66B2FF']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加数据点
        for i, data in enumerate(data_to_plot, 1):
            y = data
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, c='black')

        # 统计检验
        t_result = paired_t_test(baseline_data, our_data, metric)

        # 添加显著性标注
        y_max = max(baseline_data.max(), our_data.max())
        y_min = min(baseline_data.min(), our_data.min())
        y_range = y_max - y_min

        if t_result['significant']:
            # 显著性星号
            stars = '***' if t_result['p_value'] < 0.001 else \
                    '**' if t_result['p_value'] < 0.01 else '*'

            # 绘制显著性线
            ax.plot([1, 2], [y_max + y_range * 0.05, y_max + y_range * 0.05],
                    'k-', linewidth=1.5)
            ax.text(1.5, y_max + y_range * 0.08, stars,
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

            # 添加p值
            ax.text(1.5, y_max + y_range * 0.15,
                    f'p = {t_result["p_value"]:.4f}',
                    ha='center', va='bottom', fontsize=10)

        # 添加均值标注
        ax.hlines(baseline_data.mean(), 0.7, 1.3, colors='red',
                  linestyles='dashed', linewidth=2, label='Mean')
        ax.hlines(our_data.mean(), 1.7, 2.3, colors='red',
                  linestyles='dashed', linewidth=2)

        ax.set_ylabel(metric.upper(), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison visualization saved to {output_path}")


def run_comprehensive_statistical_test(
    baseline_results_path: Path,
    our_results_path: Path,
    output_dir: Path
):
    """
    运行完整的统计检验流程

    假设结果文件格式：
    {
        "per_sample_metrics": {
            "iou": [0.5, 0.6, 0.7, ...],
            "f1": [0.7, 0.8, 0.75, ...],
            "mae": [0.1, 0.08, 0.12, ...]
        }
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open(baseline_results_path, 'r') as f:
        baseline_data = json.load(f)

    with open(our_results_path, 'r') as f:
        our_data = json.load(f)

    # 提取逐样本指标
    baseline_metrics = baseline_data.get('per_sample_metrics', {})
    our_metrics = our_data.get('per_sample_metrics', {})

    if not baseline_metrics or not our_metrics:
        print("Warning: No per-sample metrics found. Cannot perform statistical tests.")
        print("Please ensure your evaluation scripts save per-sample results.")
        return

    # 运行检验
    results = {}
    for metric in ['iou', 'f1', 'mae', 'precision', 'recall']:
        if metric not in baseline_metrics or metric not in our_metrics:
            continue

        baseline_array = np.array(baseline_metrics[metric])
        our_array = np.array(our_metrics[metric])

        # 配对t检验
        t_result = paired_t_test(baseline_array, our_array, metric)

        # Wilcoxon检验（稳健性检查）
        w_result = wilcoxon_test(baseline_array, our_array, metric)

        results[metric] = {
            't_test': t_result,
            'wilcoxon_test': w_result
        }

        # 打印结果
        print(f"\n{'='*60}")
        print(f"Statistical Test Results for {metric.upper()}")
        print(f"{'='*60}")
        print(f"Baseline: {t_result['baseline_mean']:.4f} ± {t_result['baseline_std']:.4f}")
        print(f"Ours:     {t_result['our_mean']:.4f} ± {t_result['our_std']:.4f}")
        print(f"Improvement: {t_result['improvement']:.4f} ({t_result['improvement']/t_result['baseline_mean']*100:+.2f}%)")
        print(f"t-statistic: {t_result['t_statistic']:.4f}, p-value: {t_result['p_value']:.6f}")
        print(f"Cohen's d: {t_result['cohens_d']:.4f}")
        print(f"95% CI: [{t_result['95_ci_lower']:.4f}, {t_result['95_ci_upper']:.4f}]")
        print(f"Significant: {'YES ***' if t_result['significant'] else 'NO'}")

    # 保存结果
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    # 生成可视化
    visualize_comparison(baseline_metrics, our_metrics,
                         output_dir / "statistical_comparison.png")

    print(f"\n✓ Statistical test results saved to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument("--baseline_results", type=str, required=True,
                        help="Path to baseline results JSON with per-sample metrics")
    parser.add_argument("--our_results", type=str, required=True,
                        help="Path to our results JSON with per-sample metrics")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/analysis/statistical_tests",
                        help="Output directory")

    args = parser.parse_args()

    run_comprehensive_statistical_test(
        Path(args.baseline_results),
        Path(args.our_results),
        Path(args.output_dir)
    )
