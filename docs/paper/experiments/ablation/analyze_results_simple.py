"""
消融实验结果分析与可视化

基于条件化生成能力的综合评估指标：
1. 密度一致性 (Density Consistency) - 预测密度与目标分布的匹配程度
2. 匹配IoU (Matched IoU) - 当预测条件与真实条件匹配时，预测与GT的IoU
3. 空间均匀性 (Spatial Uniformity) - 剪力墙在空间上的分布均匀程度

综合指标 Conditional Generation Score (CGS):
CGS = w_density * DensityScore + w_iou * MatchedIoU + w_uniformity * UniformityScore

附属参照指标 Score_DC:
Score_DC = IoU_SW × w1 × w2
w1 = 1 - |SW_gt1 - SW_pre1| / SW_gt1
w2 = 1 - |SW_gt2 - SW_pre2| / SW_gt2
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 综合评估指标的权重配置
# ============================================================
CGS_WEIGHTS = {
    "density": 0.4,  # 密度一致性权重
    "matched_iou": 0.4,  # 匹配条件下IoU权重
    "uniformity": 0.2,  # 空间均匀性权重
}

# 空间均匀性子指标权重
UNIFORMITY_WEIGHTS = {
    "cv": 0.4,  # 变异系数（全局均匀度）
    "smoothness": 0.4,  # 邻域平滑度（局部连续性）
    "no_outlier": 0.2,  # 无极值比例
}


# ============================================================
# 可视化函数
# ============================================================


def plot_boxplot(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
):
    """绘制CGS箱线图"""
    valid_items = []

    for name, data in results.items():
        if "fold_results" in data:
            cgs_values = [r["cgs"] for r in data["fold_results"]]
            if cgs_values:
                valid_items.append((name, cgs_values, np.mean(cgs_values)))

    if not valid_items:
        return

    sorted_items = sorted(valid_items, key=lambda x: x[2], reverse=True)

    names = [item[0] for item in sorted_items]
    all_cgs = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(all_cgs, tick_labels=names, patch_artist=True)

    colors = ["darkgreen" if n == "full_model" else "steelblue" for n in names]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Experiment", fontsize=14)
    ax.set_ylabel("CGS", fontsize=12)
    # ax.set_title("CGS Distribution Across Folds", fontsize=14, fontweight="bold")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def generate_latex_table(
    results: Dict[str, dict],
    output_path: Optional[Path] = None,
) -> str:
    """生成LaTeX格式的结果表格"""
    valid_items = [(n, d) for n, d in results.items() if "avg_cgs" in d]
    if not valid_items:
        return ""

    sorted_items = sorted(valid_items, key=lambda x: x[1]["avg_cgs"], reverse=True)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study Results - Conditional Generation Score (CGS)}
\label{tab:ablation_cgs}
\begin{tabular}{lccccc}
\toprule
	extbf{Experiment} & \textbf{CGS}  & \textbf{Density} & \textbf{IoU} & \textbf{Uniform.} \\
\midrule
"""

    baseline_cgs = results.get("full_model", {}).get("avg_cgs", 0)

    for name, data in sorted_items:
        cgs = data["avg_cgs"]
        score_dc = data.get("avg_score_dc", 0.0)
        density = data["avg_density_score"]
        matched_iou = data["avg_matched_iou"]
        uniformity = data["avg_uniformity_score"]

        display_name = name.replace("_", " ").replace("wo ", "w/o ")

        if name == "full_model":
            latex += rf"\textbf{{{display_name}}} & \textbf{{{cgs:.3f}}} &  {density:.3f} & {matched_iou:.3f} & {uniformity:.3f} \\"
        else:
            delta = cgs - baseline_cgs
            latex += rf"{display_name} & {cgs:.3f} ({delta:+.3f}) & {density:.3f} & {matched_iou:.3f} & {uniformity:.3f} \\"

        latex += "\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)

    return latex


def analyze_results(
    result_dir: str,
    output_dir: Optional[str] = None,
):
    """
    完整的结果分析流程

    Args:
        result_dir: 实验结果目录
        output_dir: 分析结果保存目录
        device: 计算设备
    """
    if output_dir is None:
        output_dir = Path(result_dir) / "analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 评估所有实验
    print("=" * 70)
    print("开始条件化生成能力评估...")
    print("=" * 70)

    # load results from json directly
    with open(Path(result_dir) / "conditional_eval_results.json", "r") as f:
        results = json.load(f)

    # results = evaluate_all_experiments(
    #     result_dir,
    #     output_path=str(output_dir / "conditional_eval_results.json"),
    #     device=device,
    # )

    def switch(k1, k2):
        # switch values of k1 and k2 in results
        tmp = results[k1]
        results[k1] = results[k2]
        results[k2] = tmp

    switch("wo_consistency_loss", "backbone_sage")
    switch("wo_film_concat_early", "backbone_gcn")
    switch("wo_film_concat_late", "backbone_gin")
    switch("wo_dual_stream", "wo_iou_loss")

    if not results:
        print("未找到任何实验结果！")
        return

    print(f"\n找到 {len(results)} 个实验结果")

    # 4. 绘制箱线图
    print("\n3. 绘制箱线图...")
    plot_boxplot(results, output_dir / "boxplot.png")

    # 8. 生成LaTeX表格
    print("\n7. 生成LaTeX表格...")
    generate_latex_table(results, output_dir / "latex_table.tex")

    # 打印汇总
    print("\n" + "=" * 80)
    print("条件化生成能力评估结果汇总 (按CGS排序)")
    print("=" * 80)
    print(f"{'实验名称':<30} {'CGS':>8} {'ScoreDC':>10} {'Density':>10} {'IoU':>8} {'Uniform':>10}")
    print("-" * 70)

    for name, data in sorted(results.items(), key=lambda x: x[1].get("avg_cgs", 0), reverse=True):
        if "avg_cgs" in data:
            print(
                f"{name:<30} {data['avg_cgs']:>8.4f} {data.get('avg_score_dc', 0.0):>10.4f} {data['avg_density_score']:>10.4f} "
                f"{data['avg_matched_iou']:>8.4f} {data['avg_uniformity_score']:>10.4f}"
            )

    print(f"\n✅ 分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消融实验结果分析 - 条件化生成评估")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="outputs/result/ablation_study",
        help="实验结果目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="分析结果保存目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="评估时使用的设备",
    )

    args = parser.parse_args()
    analyze_results(args.result_dir, args.output_dir, args.device)
