"""
生成条件化生成密度统计的LaTeX表格

根据conditional_eval.json生成论文使用的表格
"""

import argparse
import json


def generate_density_table(json_path: str, output_path: str = None):
    """
    生成密度统计LaTeX表格

    Args:
        json_path: conditional_eval.json路径
        output_path: 输出LaTeX文件路径（可选）
    """
    # 读取JSON数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取数据
    training_stats = data["training_stats"]
    predicted_stats = data["predicted_density_stats"]
    density_mae = data["density_mae"]

    # 类别映射
    condition_names = {
        "0": "Low",
        "1": "Medium",
        "2": "High",
    }

    # 生成LaTeX表格
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Density Statistics by Condition}",
        r"\label{tab:density_stats}",
        r"\begin{tabularx}{\textwidth}{lXXX}",
        r"\toprule",
        r"Input Condition & Target Density & Generated Density (Mean $\pm$ Std) & Density Error \\",
        r"\midrule",
    ]

    # 填充数据
    for cond_id in ["0", "1", "2"]:
        target_mean = training_stats[cond_id]["mean"]
        target_std = training_stats[cond_id]["std"]

        gen_mean = predicted_stats[cond_id]["mean"]
        gen_std = predicted_stats[cond_id]["std"]

        relative_density_error = abs(target_mean - gen_mean) / target_mean  # 相对误差

        # 格式化行
        condition_name = f"{condition_names[cond_id]} ({cond_id})"
        target_density = f"{target_mean:.2f} $\\pm$ {target_std:.2f}"
        generated_density = f"{gen_mean:.2f} $\\pm$ {gen_std:.2f}"
        density_error = f"{relative_density_error:.1%}"

        line = f"{condition_name} & {target_density} & {generated_density} & {density_error} \\\\"
        latex_lines.append(line)

    # 结束表格
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
        ]
    )

    # 输出
    latex_content = "\n".join(latex_lines)
    print("\n" + "=" * 80)
    print("Generated LaTeX Table:")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)

    # 保存到文件
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        print(f"\nTable saved to: {output_path}")

    return latex_content


def generate_full_stats_table(json_path: str, output_path: str = None):
    """
    生成更详细的统计表格，包含其他指标

    Args:
        json_path: conditional_eval.json路径
        output_path: 输出LaTeX文件路径（可选）
    """
    # 读取JSON数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 类别映射
    condition_names = {
        "0": "Low",
        "1": "Medium",
        "2": "High",
    }

    # 生成LaTeX表格
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Conditional Generation Performance}",
        r"\label{tab:conditional_performance}",
        r"\begin{tabularx}{\textwidth}{lXXXX}",
        r"\toprule",
        r"Condition & Density MAE & Within Range (\%) & Constraint Violation (\%) & Sample Count \\",
        r"\midrule",
    ]

    # 填充数据
    for cond_id in ["0", "1", "2"]:
        mae = data["density_mae"][cond_id]
        within_range = data["density_within_range"][cond_id] * 100
        violation = data["constraint_violation_rate"][cond_id] * 100
        count = data["training_stats"][cond_id]["count"]

        condition_name = f"{condition_names[cond_id]} ({cond_id})"

        line = f"{condition_name} & {mae:.3f} & {within_range:.1f} & {violation:.1f} & {count} \\\\"
        latex_lines.append(line)

    # 添加总体指标
    latex_lines.extend(
        [
            r"\midrule",
            f"Ranking Accuracy & \\multicolumn{{4}}{{c}}{{{data['ranking_accuracy']:.2f}}} \\\\",
            f"Monotonicity Score & \\multicolumn{{4}}{{c}}{{{data['monotonicity_score']:.2f}}} \\\\",
        ]
    )

    # 结束表格
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
        ]
    )

    # 输出
    latex_content = "\n".join(latex_lines)
    print("\n" + "=" * 80)
    print("Generated Full Statistics LaTeX Table:")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)

    # 保存到文件
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        print(f"\nTable saved to: {output_path}")

    return latex_content


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from conditional evaluation results")
    parser.add_argument(
        "--json_path",
        type=str,
        default="outputs/result/conditional_eval/conditional_eval.json",
        help="Path to conditional_eval.json",
    )
    parser.add_argument(
        "--output_density",
        type=str,
        default="outputs/result/conditional_eval/density_table.tex",
        help="Output path for density statistics table",
    )
    parser.add_argument(
        "--output_full",
        type=str,
        default="outputs/result/conditional_eval/full_stats_table.tex",
        help="Output path for full statistics table",
    )
    parser.add_argument(
        "--table_type",
        type=str,
        default="density",
        choices=["density", "full", "both"],
        help="Which table(s) to generate",
    )

    args = parser.parse_args()

    if args.table_type in ["density", "both"]:
        generate_density_table(args.json_path, args.output_density)

    if args.table_type in ["full", "both"]:
        print("\n")
        generate_full_stats_table(args.json_path, args.output_full)


if __name__ == "__main__":
    main()
