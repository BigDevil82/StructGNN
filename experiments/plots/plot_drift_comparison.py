"""
层间位移角对比图

读取ETABS分析导出的drift结果，绘制预测方案与工程师设计方案的层间位移角对比。
4个并排子图：Case1-X, Case1-Y, Case2-X, Case2-Y
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ======================== 配置 ========================
RESULT_DIR = "result/case_study/etabs_file"

CASES = [
    {"name": "Case 1", "folder": "L27_136"},
    {"name": "Case 2", "folder": "L1L28_163"},
]

LIMIT = 1 / 1000  # 剪力墙结构层间位移角限值

OUTPUT_PATH = "result/paper_plots/drift_comparison_green.pdf"
# ======================================================


def load_drift(filepath: str) -> np.ndarray:
    """读取drift文件，返回 (n_stories, 2) 数组 [X, Y]，楼层从底到顶"""
    data = np.loadtxt(filepath)
    # 文件中楼层自高向低排列，翻转为自低向高
    return data[::-1]


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            # "mathtext.fontset": "stl",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "legend.fontsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
        }
    )

    n_cases = len(CASES)
    fig, axes = plt.subplots(1, n_cases * 2, figsize=(3.2 * n_cases * 2, 4.5), sharey=False)
    if n_cases * 2 == 1:
        axes = [axes]

    directions = ["X", "Y"]
    col_idx = 0

    for case in CASES:
        folder = os.path.join(RESULT_DIR, case["folder"])
        drift_pred = load_drift(os.path.join(folder, "drift_pred.txt"))
        drift_gt = load_drift(os.path.join(folder, "drift_gt.txt"))

        n_stories_pred = len(drift_pred) - 1  # 总楼层数（不含base=0层）
        n_stories_gt = len(drift_gt) - 1

        for d, direction in enumerate(directions):
            ax = axes[col_idx]
            col_idx += 1

            # 楼层编号从0到顶层
            stories_pred = np.arange(0, n_stories_pred + 1)
            stories_gt = np.arange(0, n_stories_gt + 1)

            # 所有行的drift数据（含0层）
            pred_vals = drift_pred[:, d]
            gt_vals = drift_gt[:, d]

            # 绘制
            ax.plot(
                pred_vals,
                stories_pred,
                "o-",
                color="#E24A33",
                label="Predicted",
                markerfacecolor="white",
                markeredgecolor="#E24A33",
                markeredgewidth=1.2,
                markersize=4,
                zorder=3,
            )
            ax.plot(
                gt_vals,
                stories_gt,
                "s-",
                color="#2E8B57",
                label="Engineer-designed",
                markerfacecolor="white",
                markeredgecolor="#2E8B57",
                markeredgewidth=1.2,
                markersize=4,
                zorder=3,
            )

            # 规范限值竖线
            ax.axvline(
                x=LIMIT,
                color="gray",
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                label=f"Code limit (1/{int(1/LIMIT)})",
            )

            ax.set_xlabel("Inter-story drift ratio")
            ax.set_ylabel("Story")

            ax.set_title(f"{case['name']} — {direction}-direction")

            # 设置Y轴为整数楼层，从0开始
            n_stories_max = max(n_stories_pred, n_stories_gt)
            ax.set_ylim(0, n_stories_max + 0.5)
            ytick_step = 2 if n_stories_max > 10 else 1
            ax.set_yticks(np.arange(0, n_stories_max + 1, ytick_step))

            # X轴从0开始
            ax.set_xlim(left=0)

            # X轴科学记数法
            ax.ticklabel_format(axis="x", style="scientific", scilimits=(-3, -3))

            ax.grid(True, alpha=0.3, linewidth=0.5)

    # 整幅图统一图例，放在顶部
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.9, edgecolor="gray", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.5)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")

    # 同时保存png
    png_path = OUTPUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
